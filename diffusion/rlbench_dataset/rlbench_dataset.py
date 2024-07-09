import datasets
import os
import numpy as np
import pickle
from natsort import natsorted

_VERSION = datasets.Version("1.0.0")

_DESCRIPTION = "RLBench data loader for GENIMA"
_HOMEPAGE = "https://genima-bot.github.io/"
_LICENSE = "MIT License"
_CITATION = "https://genima-bot.github.io/"
_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "conditioning_image": datasets.Image(),
        "text": datasets.Value("string"),
    },
)


class RLBenchConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        cameras_str = kwargs.pop("cameras", "wrist,front,left_shoulder,right_shoulder")
        self.cameras = cameras_str.split(",")
        self.image_type = kwargs.pop("image_type", "rgb_rendered")
        self.conditioning_image_type = kwargs.pop("conditioning_image_type", "rgb")
        self.data_path = kwargs.pop("data_path", "/tmp/rlbench_dataset/")
        self.variation = kwargs.pop("variation", 0)
        self.num_demos = kwargs.pop("num_demos", 50)
        self.tasks = kwargs.pop("tasks", "take_lid_off_saucepan")
        self.tiled = kwargs.pop("tiled", True)
        self.predict_future = kwargs.pop("predict_future", False)
        self.predict_future_horizon = kwargs.pop("predict_future_horizon", 20)
        super(RLBenchConfig, self).__init__(**kwargs)


class RLBenchDataset(datasets.GeneratorBasedBuilder):
    DEFAULT_CONFIG_NAME = "default"
    BUILDER_CONFIGS = [
        RLBenchConfig(
            name=DEFAULT_CONFIG_NAME,
            version=_VERSION,
            description="Genima RLBench Dataset",
        ),
    ]
    BUILDER_CONFIG_CLASS = RLBenchConfig

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data_path": self.config.data_path,
                },
            ),
        ]

    def _generate_examples(self, data_path):
        print("----- RLBench dataset config ------")
        print(f"Cameras: {self.config.cameras}")
        print(f"Image type: {self.config.image_type}")
        print(f"Conditioning image type: {self.config.conditioning_image_type}")
        print(f"Data path: {self.config.data_path}")
        print(f"Variation: {self.config.variation}")
        print(f"Number of Demos: {self.config.num_demos}")
        print(f"Tasks: {self.config.tasks}")
        print(f"Tiled: {self.config.tiled}")
        print(f"Predict future horizon: {self.config.predict_future_horizon}")
        print("------------------------------------")

        data_id = -1
        tasks = self.config.tasks.split(",")
        for task in tasks:
            # Variation and num_demos folders
            variation = f"variation{self.config.variation}"
            eps_folder = os.path.join(data_path, task, variation, "episodes")

            # Language goals
            desc_file = os.path.join(
                data_path, task, variation, "variation_descriptions.pkl"
            )
            with open(desc_file, "rb") as f:
                descriptions = pickle.load(f)

            # Choose min(num_demos_in_dataset, self.config.num_demos)
            num_demos_in_dataset = len(
                [
                    f
                    for f in os.listdir(eps_folder)
                    if os.path.isdir(os.path.join(eps_folder, f))
                ]
            )
            num_demos = min(num_demos_in_dataset, self.config.num_demos)

            # Iterate over num_demos
            selected_demos = natsorted(os.listdir(eps_folder))[:num_demos]
            for ep_path in selected_demos:
                # Tiled
                if self.config.tiled:
                    rgb_path = os.path.join(
                        eps_folder, ep_path, f"{self.config.conditioning_image_type}"
                    )
                    render_path = os.path.join(
                        eps_folder, ep_path, f"{self.config.image_type}"
                    )
                    text = "tiled perspectives of a robot "
                    f"arm executing '{np.random.choice(descriptions)}'"

                    num_render_images = (
                        len([f for f in os.listdir(render_path) if ".png" in f]) - 1
                    )

                    for i in range(num_render_images):
                        if self.config.predict_future:
                            # Future observations for SuSIE
                            future_frame_index = min(
                                i + self.config.predict_future_horizon,
                                num_render_images - 1,
                            )
                            image_path = f"{future_frame_index}.png"
                        else:
                            image_path = f"{i}.png"
                        image_path = os.path.join(render_path, image_path)
                        image = open(image_path, "rb").read()

                        conditioning_image_path = f"{i}.png"
                        conditioning_image_path = os.path.join(
                            rgb_path, conditioning_image_path
                        )
                        conditioning_image = open(conditioning_image_path, "rb").read()

                        data_id += 1

                        yield data_id, {
                            "text": text,
                            "image": {
                                "path": image_path,
                                "bytes": image,
                            },
                            "conditioning_image": {
                                "path": conditioning_image_path,
                                "bytes": conditioning_image,
                            },
                        }

                # Non-tiled
                else:
                    # Iterate over cameras
                    for camera in self.config.cameras:
                        rgb_path = os.path.join(
                            eps_folder,
                            ep_path,
                            f"{camera}_{self.config.conditioning_image_type}",
                        )
                        render_path = os.path.join(
                            eps_folder, ep_path, f"{camera}_{self.config.image_type}"
                        )
                        text = "a robot arm executing '"
                        f"{np.random.choice(descriptions)}' from {camera} perspective"

                        num_render_images = len(
                            [f for f in os.listdir(render_path) if ".png" in f]
                        )

                        for i in range(num_render_images):
                            if self.config.predict_future:
                                # Future observations for SuSIE
                                future_frame_index = min(
                                    i + self.config.predict_future_horizon,
                                    len(num_render_images) - 1,
                                )
                                image_path = f"{future_frame_index}.png"
                            else:
                                image_path = f"{i}.png"
                            image_path = os.path.join(render_path, image_path)
                            image = open(image_path, "rb").read()

                            conditioning_image_path = f"{i}.png"
                            conditioning_image_path = os.path.join(
                                rgb_path, conditioning_image_path
                            )
                            conditioning_image = open(
                                conditioning_image_path, "rb"
                            ).read()

                            data_id += 1

                            yield data_id, {
                                "text": text,
                                "image": {
                                    "path": image_path,
                                    "bytes": image,
                                },
                                "conditioning_image": {
                                    "path": conditioning_image_path,
                                    "bytes": conditioning_image,
                                },
                            }
