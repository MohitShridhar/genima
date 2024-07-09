import os
import torch
from natsort import natsorted

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import AutoencoderTiny

from agent.diffusion_agent import DiffusionAgent


class SDControlNetAgent(DiffusionAgent):
    """
    Diffusion agent with SD-Turbo and ControlNet
    """

    def __init__(self, eval_cfg):
        super().__init__(eval_cfg)

    def load_checkpoint(self):
        # Choose the last checkpoint, or the one specified
        ckpt_dirs = os.listdir(self.eval_cfg.diffusion_ckpt)
        ckpt_dirs = natsorted([d for d in ckpt_dirs if "checkpoint" in d])
        if len(ckpt_dirs) > 0:
            last_ckpt = ckpt_dirs[-1]
            controlnet_ckpt = os.path.join(
                self.eval_cfg.diffusion_ckpt, last_ckpt, "controlnet"
            )
        else:
            controlnet_ckpt = self.eval_cfg.diffusion_ckpt

        # Load ControlNet checkpoint with pre-trained Stable Diffusion
        controlnet = ControlNetModel.from_pretrained(
            controlnet_ckpt,
            torch_dtype=torch.float16,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.eval_cfg.sd_ckpt,
            controlnet=controlnet,
            safety_checker=self.eval_cfg.safety_checker,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # Set autoencoder
        if "taesd" in self.eval_cfg.autoencoder:
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                self.eval_cfg.autoencoder,
                torch_dtype=torch.float16,
            )

        # Compile, if enabled
        if self.eval_cfg.torch_compile:
            self.pipe.text_encoder = torch.compile(
                self.pipe.text_encoder, mode="reduce-overhead", fullgraph=True
            )
            self.pipe.unet = torch.compile(
                self.pipe.unet, mode="reduce-overhead", fullgraph=True
            )
            self.pipe.controlnet = torch.compile(
                self.pipe.controlnet, mode="reduce-overhead", fullgraph=True
            )

        # Channel last, if enabled
        if self.eval_cfg.channel_last:
            self.pipe.unet.to(memory_format=torch.channels_last)

    def infer(self, *args, **kwargs):
        target_images = self.pipe(
            prompt=kwargs["prompts"],
            image=kwargs["images"],
            negative_prompt=kwargs["negative_prompts"],
            num_inference_steps=kwargs["num_inference_steps"],
            guidance_scale=kwargs["guidance_scale"],
            generator=kwargs["generator"],
        )
        return target_images
