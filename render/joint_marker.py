import copy
import numpy as np
import pyrender
import os
import trimesh
from PIL import Image
from pyrender.constants import RenderFlags
from typing import List


class JointMarker:
    """
    JointMarker class to draw actions on images
    """

    def __init__(
        self,
        image_width,
        image_height,
        camera_scales,
        sphere_radius=0.008,
        znear=0.00001,
        zfar=3.0,
    ):
        """
        Initializes an instance of the JointMarker class

        Parameters:
        - image_width (int): image width
        - image_height (int): image height
        - camera_scales (List[float]): scaling of spheres per camera
        - sphere_radius (float): sphere radius
        - znear (float): distance to near clipping plane for intrinsic camera
        - zfar (float): distance to the far clipping plane for intrinsic camera

        Returns:
        - None
        """
        self._image_width = image_width
        self._image_height = image_height
        self._sphere_radius = sphere_radius
        self._znear = znear
        self._zfar = zfar

        # Initialize offscreen renderer
        self._offscreen_renderer = pyrender.OffscreenRenderer(
            self._image_width, self._image_height
        )

        # Cache meshes and materials
        self.mesh_cache = {}  # camera_scale: (sphere_mesh, sphere_material)
        for camera_scale in camera_scales:
            per_cam_cache_dict = {}
            sphere_mesh = trimesh.creation.uv_sphere(
                radius=self._sphere_radius * camera_scale
            )
            sphere_material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0)
            per_cam_cache_dict["sphere"] = (sphere_mesh, sphere_material)
            self.mesh_cache[camera_scale] = per_cam_cache_dict

    def render_action(
        self,
        cam_intrinsic,
        cam_extrinsic,
        joint_matrices,
        joint_opens,
        camera_scale=1.0,
        sphere_colors=List[None],
    ):
        """
        Renders an action in a pyrender scene

        Parameters:
        - cam_intrinsic (np.array): camera intrinsics
        - cam_extrinsic (np.array): camera extrinsics
        - joint_matrices (List[np.array]): list of joint pose matrices
        - joint_opens (List[int]): list of joint open/close states \
            (only relevant for gripper joints)
        - camera_scale (int): camera scaling for sphere
        - sphere_colors (List[string]): colors for spheres

        Returns:
        - rendered_img: (h, w, 3) uint8 array
        """

        # Create a PyRender scene
        scene = pyrender.Scene()

        # Camera extrinsics
        cam_extrinsic = np.array(cam_extrinsic)

        # Create an intrinsic camera
        camera = pyrender.IntrinsicsCamera(
            fx=cam_intrinsic[0, 0],
            fy=cam_intrinsic[1, 1],
            cx=cam_intrinsic[0, 2],
            cy=cam_intrinsic[1, 2],
            znear=self._znear,
            zfar=self._zfar,
        )
        rotation_matrix = cam_extrinsic[:3, :3]

        # Define the rotation angle in radians (180 degrees)
        angle_degrees = -180
        angle_radians = np.radians(angle_degrees)
        rotation_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle_radians), -np.sin(angle_radians)],
                [0, np.sin(angle_radians), np.cos(angle_radians)],
            ]
        )

        # Perform the rotation by multiplying the matrices
        rotated_rotation_matrix = np.dot(rotation_matrix, rotation_x)

        # Update the original 4x4 matrix with the rotated 3x3 rotation matrix
        cam_extrinsic[:3, :3] = rotated_rotation_matrix

        # Add the camera to the scene
        _ = scene.add(camera, pose=cam_extrinsic)

        # Draw spheres
        for i, (gripper_matrix, gripper_open) in enumerate(
            zip(joint_matrices, joint_opens)
        ):
            # Draw sphere
            if sphere_colors[i] is None:
                sphere_texture_name = (
                    "sphere_yellow_stripe_texture.png"
                    if gripper_open <= 0.1
                    else "sphere_cyan_stripe_texture.png"
                )
            else:
                color_mapping = {
                    "green": "sphere_green_stripe_texture.png",
                    "red": "sphere_red_stripe_texture.png",
                    "purple": "sphere_purple_stripe_texture.png",
                }
                sphere_texture_name = color_mapping[sphere_colors[i]]

            sphere_texture = pyrender.Texture(
                source=Image.open(
                    os.path.join("./sphere_textures/", sphere_texture_name)
                ).convert("RGBA"),
                source_channels="RGBA",
            )

            sphere_mesh, sphere_material = self.mesh_cache[camera_scale]["sphere"]
            sphere_mesh, sphere_material = copy.deepcopy(sphere_mesh), copy.deepcopy(
                sphere_material
            )
            texture_image = np.ones((1, 1, 3), dtype=np.uint8) * 255
            texture = trimesh.visual.texture.TextureVisuals(
                uv=(
                    sphere_mesh.vertices[:, :2]
                    - np.min(sphere_mesh.vertices[:, :2], axis=0)
                )
                / np.ptp(sphere_mesh.vertices[:, :2], axis=0),
                image=texture_image,
            )
            sphere_mesh._visual = texture
            sphere_material.baseColorTexture = sphere_texture

            # Add base color factor
            # NOTE: this is not necessary, but kept for
            # reproducing original sphere colors from the paper
            base_color = (
                tuple([0.60392156862, 0.86274509803, 1.0, 1.0])
                if gripper_open > 0.1
                else tuple([1.0, 1.0, 0.0, 1.0])
            )
            sphere_material.baseColorFactor = base_color

            sphere = pyrender.Mesh.from_trimesh(sphere_mesh, material=sphere_material)
            _ = scene.add(sphere, pose=gripper_matrix)

        # Render the scene in offscreen mode
        flags = RenderFlags.FLAT
        rendered_img, _ = self._offscreen_renderer.render(scene, flags)
        return rendered_img
