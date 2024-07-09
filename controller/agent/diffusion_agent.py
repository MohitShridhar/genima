import torch
from torchvision import transforms


class DiffusionAgent:
    """
    Base class for diffusion agents
    """

    def __init__(self, eval_cfg):
        self.eval_cfg = eval_cfg
        self.pipe = None

        self.load_checkpoint()
        self.set_optimizations()
        self.common_setup()

    def load_checkpoint(self):
        raise NotImplementedError()

    def set_optimizations(self):
        if self.eval_cfg.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if self.eval_cfg.vae_slicing:
            self.pipe.vae.enable_slicing()

        if self.eval_cfg.upcast_vae:
            self.pipe.upcast_vae()

        if self.eval_cfg.fused_projections:
            self.pipe.fuse_qkv_projections(vae=False)

        if self.eval_cfg.enable_xformers_memory_efficient_attention:
            self.pipe.enable_xformers_memory_efficient_attention()

        self.pipe.set_progress_bar_config(
            disable=(not self.eval_cfg.show_diffusion_progress)
        )

        self.pipe.to(self.eval_cfg.device)

    def common_setup(self):
        resolution = self.eval_cfg.image_resolution
        self.transform_to_resolution = transforms.Compose(
            [
                transforms.Resize(
                    resolution, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(resolution),
            ]
        )

        self.transform_to_half_resolution = transforms.Compose(
            [
                transforms.Resize(
                    resolution // 2, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(resolution // 2),
            ]
        )

    def infer(self, *args, **kwargs):
        raise NotImplementedError()
