import numpy as np
import torch
from PIL import Image


def tile_images(rgbs, num_frames):
    assert type(rgbs[0]) == Image.Image, "Images must be PIL Images"
    assert rgbs[0].size == (256, 256), "For tiling, image sizes must be 256x256"

    tiled_images = []
    for t in range(num_frames):
        tiled_image = Image.new("RGB", (512, 512))
        tiled_image.paste(rgbs[0 * num_frames + t], (0, 0))
        tiled_image.paste(rgbs[1 * num_frames + t], (256, 0))
        tiled_image.paste(rgbs[2 * num_frames + t], (0, 256))
        tiled_image.paste(rgbs[3 * num_frames + t], (256, 256))
        tiled_images.append(tiled_image)

    return tiled_images


def untile_images(gen_images, cameras, resize_transform):
    assert gen_images[0].size == (512, 512), "For untiling, image sizes must be 512x512"

    crop_order = [
        (0, 0, 256, 256),
        (256, 0, 512, 256),
        (0, 256, 256, 512),
        (256, 256, 512, 512),
    ]

    untiled_images = {camera: [] for camera in cameras}
    for t in range(len(gen_images)):
        tiled_image = gen_images[t]

        for cam_idx, camera in enumerate(cameras):
            crop = crop_order[cam_idx]
            gen_image = resize_transform(tiled_image.crop(crop))
            gen_image = np.transpose(
                np.expand_dims(np.array(gen_image), axis=0), (0, 3, 1, 2)
            )
            untiled_images[camera].append(gen_image)

    for camera in cameras:
        untiled_images[camera] = np.concatenate(untiled_images[camera], axis=0)

    return untiled_images


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return (
            tensor
            + torch.randn(tensor.size(), device=tensor.device) * self.std
            + self.mean
        )

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )
