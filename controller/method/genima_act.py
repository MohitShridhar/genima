from typing import Tuple, List, Optional
from robobase.method.act import ActBCAgent, ACTPolicy
import numpy as np
import torch
import torch.nn as nn
from robobase.replay_buffer.replay_buffer import ReplayBuffer
from robobase.method.utils import (
    extract_from_spec,
    extract_from_batch,
    flatten_time_dim_into_channel_dim,
    stack_tensor_dictionary,
    extract_many_from_batch,
)
from robobase.models.multi_view_transformer import (
    MultiViewTransformerEncoderDecoderACT,
    reparametrize,
)
from robobase.models.act.utils.misc import kl_divergence
import torchvision.transforms as transforms
from torchvision.transforms import v2
import torch.nn.functional as F

from utils.misc import AddGaussianNoise


class GenimaMVTransformer(MultiViewTransformerEncoderDecoderACT):
    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
        qpos: torch.Tensor,
        actions: torch.Tensor = None,
        is_pad: torch.Tensor = None,
        task_emb: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the Genima Multi View Transformer model.

        Args:
            x (Tuple[torch.Tensor, torch.Tensor]):
                    Image features and positional encodings.
            qpos (torch.Tensor): Tensor containing proprioception features.
            actions (torch.Tensor, optional): Tensor containing action sequences.
            is_pad (torch.Tensor, optional): Tensor indicating padding positions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
                    Tuple containing action predictions,
                    padding predictions,
                    and a list of latent variables [mu, logvar].
        """

        bs = x[0].shape[0]

        # Proprioception features
        proprio_input = self.input_proj_robot_state(qpos)

        if self.training and actions is not None:
            actions = actions[:, : self.num_queries]
            is_pad = is_pad[:, : self.num_queries]

            # Compress action and qpos into style variable: latent_input
            encoder_output = self.style_variable_encoder(bs, actions, qpos, is_pad)

            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)

        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(
                qpos.device
            )
            latent_input = self.latent_out_proj(latent_sample)

        # Apply transformer block
        hs = self.transformer(
            x[0],
            None,
            self.query_embed.weight,
            x[1],
            latent_input,
            proprio_input,
            self.additional_pos_embed.weight,
            task_emb=task_emb,
        )[-1]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)

        return a_hat, is_pad_hat, [mu, logvar]

    def calculate_loss(
        self,
        input_feats: Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]],
        actions: torch.Tensor,
        is_pad: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, dict]]:
        """
        Calculate the loss for the MultiViewTransformerEncoderDecoderACT model.

        Args:
            input_feats (Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]):
                    Tuple containing action predictions, padding predictions,
                    and a list of latent variables [mu, logvar].
            actions (torch.Tensor): Tensor containing ground truth action sequences.
            is_pad (torch.Tensor): Tensor indicating padding positions.

        Returns:
            Optional[Tuple[torch.Tensor, dict]]:
                    Tuple containing the loss tensor and a dictionary of loss
                    components.
        """
        a_hat = input_feats[0]
        mu, logvar = input_feats[2]

        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        loss_dict = dict()
        all_l1 = F.l1_loss(actions[..., :-1], a_hat[..., :-1], reduction="none")
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        gripper_loss = (
            F.binary_cross_entropy_with_logits(
                a_hat[..., -1], actions[..., -1], reduction="none"
            )
            * 0.05
        )
        gripper_loss = (gripper_loss * ~is_pad).mean()

        loss_dict["l1"] = l1
        loss_dict["gripper_loss"] = gripper_loss
        loss_dict["kl"] = total_kld[0]
        loss_dict["loss"] = (
            loss_dict["l1"]
            + loss_dict["gripper_loss"]
            + loss_dict["kl"] * self.kl_weight
        )

        return (loss_dict["loss"], loss_dict)


class GenimaACTPolicy(ACTPolicy):
    def __init__(self, data_augmentation, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.data_augmentation = data_augmentation
        self.aug_transforms = transforms.Compose(
            [
                v2.RandomApply([v2.ElasticTransform(alpha=80.0, sigma=10.0)]),
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
                        )
                    ]
                ),
                v2.RandomApply([v2.RandomCrop(size=(256, 256), padding=4)]),
                AddGaussianNoise(0, 5.0),
            ]
        )

    def forward(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        actions: torch.Tensor = None,
        is_pad: torch.Tensor = None,
        task_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            qpos (torch.Tensor): joint positions.
            image (torch.Tensor): Input image data (unnormalized).
            actions (torch.Tensor, optional): Actions. Default is None.
            is_pad (torch.Tensor, optional): Whether actions are padded.
                Default is None.
            task_emb: task instruction embeddings. Default is None
        """

        if actions is not None:
            image = self.aug_transforms(image.to(dtype=torch.uint8))

        image = self.normalize(image / 255.0)

        feat, pos, task_emb = self.encoder_model(image, task_emb)
        # feat: (b, fs * hidden_dim, 3, 3*v) -> (b, hidden_dim, 3, 3*v)
        # NOTE: the detr_vae used by ACT expects views to be on the width channel.
        if self.frame_stack > 1:
            x = (
                self.projection_layer(feat),
                pos,
            )  # pass through projection layer to reduce the channel dimension
        else:
            x = (feat, pos)  # If frame_stack == 1, directly use the raw feature.

        if actions is not None:
            x = self.actor_model(
                x, qpos, actions=actions, is_pad=is_pad, task_emb=task_emb
            )
            loss, loss_dict = self.actor_model.calculate_loss(
                x, actions=actions, is_pad=is_pad
            )
            return loss_dict

        else:
            x = self.actor_model(
                x, qpos, actions=actions, is_pad=is_pad, task_emb=task_emb
            )
            return x[0]


class GenimaACT(ActBCAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_actor(self):
        # NOTE: Encoder returns visual_obs_feat, pos_emb, task_emb, we pass
        # visual_obs_feat shape into actor model constructor
        data_augmentation = self.actor_model.keywords.pop("data_augmentation")

        self.actor_model = self.actor_model(
            input_shape=self.encoder.output_shape[0],
            state_dim=np.prod(self.observation_space["low_dim_state"].shape),
            action_dim=self.action_space.shape[-1],
        ).to(self.device)

        # We use an extra layer to project robot state
        state_dim, hidden_dim = (
            self.actor_model.input_proj_robot_state.in_features,
            self.actor_model.input_proj_robot_state.out_features,
        )
        self.actor_model.input_proj_robot_state = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rgb_latent_size = self.actor_model.output_shape[-1]

        self.actor = GenimaACTPolicy(
            data_augmentation=data_augmentation,
            observation_space=self.observation_space,
            actor_model=self.actor_model,
            encoder_model=self.encoder,
        ).to(self.device)

        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.actor.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.actor.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.lr_backbone,
            },
        ]

        self.actor_opt = torch.optim.AdamW(
            param_dicts, lr=self.lr, weight_decay=self.weight_decay
        )

    def act(self, obs, step, eval_mode: bool) -> torch.Tensor:
        """
        Perform an action given observations.

        Args:
            obs (dict): Dictionary containing observations.
            step (int): The current step.
            eval_mode (bool): If True, act in evaluation mode;
                    otherwise, act in training mode.

        Returns:
            np.ndarray: The agent's action.

        """

        if self.low_dim_size > 0:
            qpos = flatten_time_dim_into_channel_dim(
                extract_from_spec(obs, "low_dim_state")
            )
            qpos = qpos.detach()

        if self.use_pixels:
            rgb = flatten_time_dim_into_channel_dim(
                stack_tensor_dictionary(extract_many_from_batch(obs, r"rgb.*"), 1),
                has_view_axis=False,
            )

            image = rgb.float().detach()
            image = image.view(image.shape[0], -1, 3, image.shape[-2], image.shape[-1])

        task_emb = None
        if self.actor.encoder_model.use_lang_cond:
            lang_tokens = flatten_time_dim_into_channel_dim(
                extract_from_spec(obs, "lang_tokens")
            )
            task_emb, _ = self.encode_clip_text(lang_tokens)

        action = self.actor(qpos, image, task_emb=task_emb)

        return action

    def encode_clip_text(self, tokens):
        if hasattr(self, "clip_model") is False:
            # load CLIP model
            print("Loading CLIP model...")
            import clip

            self.clip_model, _ = clip.load("ViT-B/32")
            del self.clip_model.visual

        with torch.no_grad():
            dtype = torch.float16
            token_shape = tokens.shape
            tks = tokens.view(-1, tokens.shape[-1])
            x = self.clip_model.token_embedding(tks).type(
                dtype
            )  # [batch_size, n_ctx, d_model]

            x = x + self.clip_model.positional_embedding.type(dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_model.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.clip_model.ln_final(x).type(dtype)

            emb = x.clone()
            # x.shape = [batch_size, n_ctx, transformer.width]
            x = (
                x[torch.arange(x.shape[0]), tks.argmax(dim=-1)]
                @ self.clip_model.text_projection
            )

            x = x.view(token_shape[0], token_shape[1], -1)
            x = x[:, 0].to(torch.float32)  # text shouldnt change across two frames
        return x, emb

    def update(
        self, replay_iter, step: int, replay_buffer: ReplayBuffer = None
    ) -> dict:
        """
        Update the agent's policy using behavioral cloning.

        Args:
            replay_iter (iterable): An iterator over a replay buffer.
            step (int): The current step.
            replay_buffer (ReplayBuffer): The replay buffer.

        Returns:
            dict: Dictionary containing training metrics.

        """

        metrics = dict()
        batch = next(replay_iter)
        batch = {k: torch.tensor(v).to(self.device) for k, v in batch.items()}
        actions = batch["action"]
        reward = batch["reward"]

        if self.low_dim_size > 0:
            obs = flatten_time_dim_into_channel_dim(
                extract_from_batch(batch, "low_dim_state")
            )
            qpos = obs.detach()

        rgb = flatten_time_dim_into_channel_dim(
            # Don't get "tp1" obs
            stack_tensor_dictionary(
                extract_many_from_batch(batch, r"rgb(?!.*?tp1)"), 1
            ),
            has_view_axis=True,
        )
        image = rgb.float().detach()
        image = image.view(image.shape[0], -1, 3, image.shape[-2], image.shape[-1])

        task_emb = None
        if self.actor.encoder_model.use_lang_cond:
            lang_tokens = flatten_time_dim_into_channel_dim(
                extract_from_spec(batch, "lang_tokens")
            )
            task_emb, _ = self.encode_clip_text(lang_tokens)

        is_pad = torch.zeros_like(actions)[:, :, 0].bool()
        loss_dict = self.actor(
            qpos, image, actions=actions, is_pad=is_pad, task_emb=task_emb
        )
        if self.use_pixels and self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
            if self.use_multicam_fusion and self.view_fusion_opt is not None:
                self.view_fusion_opt.zero_grad(set_to_none=True)
        self.actor_opt.zero_grad(set_to_none=True)
        loss_dict["loss"].backward()
        if self.actor_grad_clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        self.actor_opt.step()
        if self.use_pixels and self.encoder is not None:
            if self.actor_grad_clip:
                nn.utils.clip_grad_norm_(
                    self.encoder.parameters(), self.critic_grad_clip
                )
            self.encoder_opt.step()
            if self.use_multicam_fusion and self.view_fusion_opt is not None:
                self.view_fusion_opt.step()

        if self.logging:
            metrics["actor_loss"] = loss_dict["loss"].item()
            metrics["actor_l1_loss"] = loss_dict["l1"].item()
            metrics["actor_gripper_loss"] = loss_dict["gripper_loss"].item()
            metrics["actor_kl_loss"] = loss_dict["kl"].item()
            metrics["batch_reward"] = reward.mean().item()

        return metrics
