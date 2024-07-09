import hydra
import os
import numpy as np
from tqdm import tqdm
import multiprocessing
import pickle
from PIL import Image
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
import shutil
from joint_marker import JointMarker

os.environ["PYOPENGL_PLATFORM"] = "egl"

JOINT_COLOR_MAP = {
    1: "red",
    3: "green",
    5: "purple",
}


class RenderData:
    """
    RenderData class to draw actions in images of a RLBench dataset
    """

    def __init__(self, cfg) -> None:
        """
        Initializes an instance of RenderData class

        Parameters:
        - cfg (DictConfig): configuration for RenderData

        Returns:
        - None
        """
        self.cfg = cfg
        # Initialize texture files
        textures_path = cfg.textures_path
        self._texture_files = [
            os.path.join(textures_path, f) for f in os.listdir(textures_path)
        ]
        self._image_width = cfg.image_width
        self._image_height = cfg.image_height

        # Create new directory to save if specfied
        # Otherwise, modify based on existing directory name
        dataset_root = self.cfg.dataset_root
        full_dataset_root = os.path.abspath(dataset_root)
        full_dataset_root = (
            full_dataset_root[:-1]
            if full_dataset_root[-1] == "/"
            else full_dataset_root
        )
        dataset_dir_name = full_dataset_root.split("/")[-1]
        self._full_context_dataset_dir_name = dataset_dir_name + "_rgb_rendered"
        self._random_context_dataset_dir_name = dataset_dir_name + "_rnd_bg"

        if self.cfg.save_path is not None:
            self._parent_path = self.cfg.save_path
        else:
            self._parent_path = os.path.dirname(full_dataset_root)

        # Copy data from the original directory
        if self.cfg.draw.rgb_rendered:
            self._full_context_dst_path = os.path.join(
                self._parent_path, self._full_context_dataset_dir_name
            )
            self.ensure_folder_exists(self._full_context_dst_path)
            self.deepcopy_folder(
                os.path.join(full_dataset_root, self.cfg.task),
                os.path.join(self._full_context_dst_path, self.cfg.task),
            )

        if self.cfg.draw.rnd_bg:
            self.ensure_folder_exists(
                os.path.join(self._parent_path, self._random_context_dataset_dir_name)
            )
            self._random_context_dst_path = os.path.join(
                self._parent_path, self._random_context_dataset_dir_name
            )
            self.deepcopy_folder(
                os.path.join(full_dataset_root, self.cfg.task),
                os.path.join(self._random_context_dst_path, self.cfg.task),
            )

    def tile_images(self, image_list, output_path):
        """
        Combine a list of images to a single tiled image

        Parameters:
        - image_list (List[PIL.Image]): list of images to be tiled
        - output_path (string): path to save the tiled image

        Returns:
        - None
        """
        # Assuming image_list contains four PIL images
        if len(image_list) != 4:
            raise ValueError("The input list must contain exactly four PIL images.")

        # Get the dimensions of the original images
        width, height = image_list[0].size

        # Calculate the dimensions of each tile
        tile_width = width
        tile_height = height

        # Create a new image with double the width and height
        tiled_image = Image.new("RGB", (width * 2, height * 2))

        # Paste the images into the 2x2 grid
        tiled_image.paste(image_list[0], (0, 0))
        tiled_image.paste(image_list[1], (tile_width, 0))
        tiled_image.paste(image_list[2], (0, tile_height))
        tiled_image.paste(image_list[3], (tile_width, tile_height))

        # Save the resulting image
        tiled_image.save(output_path)

    def render_demo(self, episode):
        """
        Renders actions for a demonstration

        Parameters:
        - episode (int): episode index

        Returns:
        - None
        """
        action_marker = JointMarker(
            image_width=self.cfg.image_width,
            image_height=self.cfg.image_height,
            camera_scales=self.cfg.camera_scales,
            sphere_radius=self.cfg.render.sphere.radius,
            znear=self.cfg.znear,
            zfar=self.cfg.zfar,
        )
        cameras = self.cfg.cameras

        assert os.path.exists(self.cfg.dataset_root)
        dataset_root = self.cfg.dataset_root

        # Create tiled directories
        if self.cfg.draw.rgb_rendered:
            cfg_file = os.path.join(
                self._full_context_dst_path, "render_data_config.yaml"
            )
            with open(cfg_file, "w") as f:
                OmegaConf.save(config=self.cfg, f=f)

            # Create tiled png folders
            rgb_rendered_episode_path = os.path.join(
                self._full_context_dst_path,
                self.cfg.task,
                (
                    f"variation{self.cfg.variation}"
                    if self.cfg.variation != -1
                    else "all_variations"
                ),
                "episodes",
                f"episode{episode}",
            )
            self.make_png_folder(rgb_rendered_episode_path, "", suffix="tiled_rgb")
            self.make_png_folder(
                rgb_rendered_episode_path, "", suffix="tiled_rgb_rendered"
            )
            low_dim_obs_path = os.path.join(
                rgb_rendered_episode_path, "low_dim_obs.pkl"
            )

        # Create tiled directories
        if self.cfg.draw.rnd_bg:
            cfg_file = os.path.join(
                self._random_context_dst_path, "render_data_config.yaml"
            )
            with open(cfg_file, "w") as f:
                OmegaConf.save(config=self.cfg, f=f)

            # Create tiled png folders
            rnd_bg_episode_path = os.path.join(
                self._random_context_dst_path,
                self.cfg.task,
                (
                    f"variation{self.cfg.variation}"
                    if self.cfg.variation != -1
                    else "all_variations"
                ),
                "episodes",
                f"episode{episode}",
            )
            low_dim_obs_path = os.path.join(rnd_bg_episode_path, "low_dim_obs.pkl")

        og_episode_path = os.path.join(
            dataset_root,
            self.cfg.task,
            (
                f"variation{self.cfg.variation}"
                if self.cfg.variation != -1
                else "all_variations"
            ),
            "episodes",
            f"episode{episode}",
        )
        with open(low_dim_obs_path, "rb") as f:
            low_dim_obs = pickle.load(f)

        rgb_imgs = {}

        for camera in cameras:
            rgb_path = os.path.join(og_episode_path, f"{camera}_rgb")
            rgbs = []
            for i in range(len(low_dim_obs)):
                img = Image.open(os.path.join(rgb_path, f"{i}.png"))
                rgbs.append(np.array(img))
            rgb_imgs[f"{camera}_rgb"] = rgbs

            assert len(low_dim_obs) == len(rgbs)

        for ts in tqdm(range(len(low_dim_obs) - 1)):
            rgbs, rgb_renders = [], []
            for c_idx, camera in enumerate(cameras):
                curr_intrinsic = np.array(
                    low_dim_obs[ts].misc[f"{camera}_camera_intrinsics"]
                )
                curr_extrinsic = np.array(
                    low_dim_obs[ts].misc[f"{camera}_camera_extrinsics"]
                )
                rgb = rgb_imgs[f"{camera}_rgb"][ts]
                if "overhead" not in camera:
                    rgbs.append(Image.fromarray(rgb))

                joint_matrices, joint_opens, colors = [], [], []

                last_idx = min(ts + 1 + self.cfg.action_horizon, len(low_dim_obs) - 1)
                for _, i in enumerate(range(ts + 1, last_idx)):
                    obs_idx = i

                    gripper_matrix = np.array(low_dim_obs[obs_idx].gripper_matrix)
                    gripper_open = low_dim_obs[obs_idx].gripper_open

                    if obs_idx == last_idx - 1:
                        joint_matrices = [gripper_matrix]
                        joint_opens = [gripper_open]
                        colors = [
                            None
                        ]  # None indicates that gripper joint color should be determined by gripper_open

                        for joint in self.cfg.render.joints[camera]:
                            joint_pose = low_dim_obs[obs_idx].misc["joint_poses"][joint]
                            joint_mat = np.eye(4)
                            joint_mat[:3, 3] = joint_pose[:3]
                            joint_rot = Rotation.from_quat(
                                [
                                    joint_pose[3],
                                    joint_pose[4],
                                    joint_pose[5],
                                    joint_pose[6],
                                ]
                            )
                            joint_mat[:3, :3] = joint_rot.as_matrix()

                            joint_matrices.append(joint_mat)
                            joint_opens.append(1.0)
                            colors.append(JOINT_COLOR_MAP[joint])

                    colors.append(None)

                # Draw actions with full-context background
                if self.cfg.draw.rgb_rendered:
                    render = action_marker.render_action(
                        curr_intrinsic,
                        curr_extrinsic,
                        joint_matrices,
                        joint_opens,
                        camera_scale=self.cfg.camera_scales[c_idx],
                        sphere_colors=colors,
                    )
                    render = np.array(render)
                    render_rnd_bg = np.array(render)

                    white_space = np.all(render == [255, 255, 255], axis=-1)
                    occupied_space = np.any(render != [255, 255, 255], axis=-1)
                    render[white_space] = rgb[white_space]

                    pil_img = Image.fromarray(render)
                    pil_img.save(
                        os.path.join(
                            rgb_rendered_episode_path, f"{camera}_rgb", f"{ts}.png"
                        )
                    )

                    if "overhead" not in camera:
                        rgb_renders.append(pil_img.copy())

                # Draw actions with random background
                if self.cfg.draw.rnd_bg:
                    texture_image = Image.open(np.random.choice(self._texture_files))
                    texture_image = texture_image.resize(
                        (self._image_width, self._image_height)
                    )
                    texture_image = np.array(texture_image)
                    render_rnd_bg[white_space] = texture_image[white_space]
                    blend = np.random.uniform(self.cfg.alpha_blend, 1.0)
                    render_rnd_bg[occupied_space] = render_rnd_bg[
                        occupied_space
                    ] * blend + texture_image[occupied_space] * (1 - blend)
                    pil_img = Image.fromarray(render_rnd_bg)
                    pil_img.save(
                        os.path.join(rnd_bg_episode_path, f"{camera}_rgb", f"{ts}.png")
                    )

            if self.cfg.draw.rgb_rendered:
                self.tile_images(
                    rgbs,
                    os.path.join(rgb_rendered_episode_path, "tiled_rgb", f"{ts}.png"),
                )
                self.tile_images(
                    rgb_renders,
                    os.path.join(
                        rgb_rendered_episode_path, "tiled_rgb_rendered", f"{ts}.png"
                    ),
                )

    def generate(self):
        """
        Parent function to render actions for a list of RLBench demonstrations
        """
        for episode in tqdm(range(self.cfg.episode_offset, self.cfg.episodes)):
            self.render_demo(episode)

    def make_png_folder(self, episode_path, camera, suffix="_rgb_rendered"):
        """
        Create png folder with the given suffix if it doesn't exist.
        If the folder exists, removes all the png files inside of it

        Parameters:
        - episode_path (string): path of an episode
        - camera (string): camera name
        - suffix (string): suffix to be added to directory name

        Returns:
        - None
        """
        png_dir_path = os.path.join(episode_path, f"{camera}{suffix}")
        if self.ensure_folder_exists(png_dir_path):
            # delete existing pngs
            for file in os.listdir(png_dir_path):
                if file.endswith(".png"):
                    os.remove(os.path.join(episode_path, f"{camera}{suffix}", file))

    def ensure_folder_exists(self, folder_path):
        """
        Checks if the directory exists for the given path.
        Creates the folder if it doesn't.

        Parameters:
        - folder_path (string): directory path

        Returns:
        - None
        """

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return False
        return True

    def deepcopy_folder(self, src_folder, dst_folder):
        """
        Creates a deepcopy of a given folder path

        Parameters:
        - src_folder (string): source directory path
        - dst_folder (string): destination directory path

        Returns:
        - None
        """
        # Check if the source folder exists
        if not os.path.exists(src_folder):
            raise FileNotFoundError(f"The source folder '{src_folder}' does not exist.")

        # Check if the destination folder exists, if not, create it
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)

        # Filter out depth and mask folders
        def ignore_files(directory, contents):
            # Filter out directories that contain 'depth' or 'mask' in their names
            ignored = [name for name in contents if "depth" in name or "mask" in name]
            return ignored

        # Use shutil.copytree to copy the directory and its contents
        shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True, ignore=ignore_files)


class RenderDataParallel(RenderData):
    """
    Child class of RenderData to enable parallel generation
    """

    def __init__(self, cfg) -> None:
        """
        Initializes an instance of RenderDataParallel class

        Parameters:
        - cfg (DictConfig): configuration for RenderData

        Returns:
        - None
        """
        super().__init__(cfg)
        self.num_processes = cfg.num_processes

    def generate(self):
        """
        Parent function to render actions for a list of RLBench demonstrations
        using multiprocessing
        """
        print(f"Number of processes used : {self.cfg.num_processes}")
        pool = multiprocessing.Pool(processes=self.cfg.num_processes)

        # Use the pool's map function to
        # apply the process_function to each object in parallel
        demo_indices = [
            idx for idx in range(self.cfg.episode_offset, self.cfg.episodes)
        ]
        _ = pool.map(self.render_demo, demo_indices)

        # Close the pool to free up resources
        pool.close()
        pool.join()


@hydra.main(config_path="cfgs", config_name="render", version_base=None)
def render_data(cfg):
    """
    Action rendering entrypoint
    """
    print(cfg)

    import time

    render_start = time.time()
    render_data = RenderDataParallel(cfg)
    render_data.generate()
    render_end = time.time()
    print(f"Render time: {render_end - render_start}")


if __name__ == "__main__":
    render_data()
