import os
import torch
from natsort import natsorted

from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers import UNet2DConditionModel

from agent.diffusion_agent import DiffusionAgent


class SDPix2PixAgent(DiffusionAgent):
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
            pix2pix_ckpt = os.path.join(self.eval_cfg.diffusion_ckpt, last_ckpt)
        else:
            pix2pix_ckpt = self.eval_cfg.diffusion_ckpt

        unet = UNet2DConditionModel.from_pretrained(
            pix2pix_ckpt,
            subfolder="unet",
            torch_dtype=torch.float16,
        )

        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.eval_cfg.sd_ckpt,
            unet=unet,
            variant="fp16",
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
