from typing import List

from rlbench.backend.observation import Observation
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from tiger.envs.rlbench import ROBOT_STATE_KEYS
from tiger.envs.rlbench import TASK_TO_LOW_DIM_SIM, _get_cam_observation_elements
from tiger.envs.rlbench import ActionModeType
from tiger.utils import DemoStep
from tiger.envs.env import DemoEnv
from tiger.replay_buffer.replay_buffer import ReplayBuffer

import numpy as np
from gymnasium import spaces
import clip


def _convert_rlbench_demos_for_loading(
    raw_demos, observation_config, desc, robot_state_keys: dict = None
) -> List[List[DemoStep]]:
    """Converts demos generated in rlbench to the common DemoStep format.

    Args:
        raw_demos: raw demos generated with rlbench.

    Returns:
        List[List[DemoStep]]: demos converted to DemoSteps ready for
            augmentation and loading.
    """
    converted_demos = []
    for demo in raw_demos:
        converted_demo = []
        for timestep in demo:
            timestep.misc["descriptions"] = desc
            converted_demo.append(
                DemoStep(
                    timestep.joint_positions,
                    timestep.gripper_open,
                    _extract_obs(timestep, observation_config, robot_state_keys),
                    timestep.gripper_matrix,
                    timestep.misc,
                )
            )
        converted_demos.append(converted_demo)
    return converted_demos


def observations_to_action_with_onehot_gripper(
    current_observation: DemoStep,
    next_observation: DemoStep,
    action_space: spaces.Box,
):
    """Calculates the action linking two sequential observations.

    Args:
        current_observation (DemoStep): the observation made before the action.
        next_observation (DemoStep): the observation made after the action.
        action_space (Box): the action space of the unwrapped env.

    Returns:
        np.ndarray: action taken at current observation. Returns None if action
            outside action_space.
    """
    action = np.concatenate(
        [
            (
                next_observation.misc["joint_position_action"][:-1]
                if "joint_position_action" in next_observation.misc
                else next_observation.joint_positions
            ),
            [1.0 if next_observation.gripper_open == 1 else 0.0],
        ]
    ).astype(np.float32)

    if np.any(action[:-1] > action_space.high[:-1]) or np.any(
        action[:-1] < action_space.low[:-1]
    ):
        return None
    return action


def _observation_config_to_gym_space(observation_config, task_name: str) -> spaces.Dict:
    space_dict = {}
    robot_state_len = 0
    if observation_config.joint_velocities:
        robot_state_len += 7
    if observation_config.joint_positions:
        robot_state_len += 7
    if observation_config.joint_forces:
        robot_state_len += 7
    if observation_config.gripper_open:
        robot_state_len += 1
    if observation_config.gripper_pose:
        robot_state_len += 7
    if observation_config.gripper_joint_positions:
        robot_state_len += 2
    if observation_config.gripper_touch_forces:
        robot_state_len += 2
    if observation_config.task_low_dim_state:
        robot_state_len += TASK_TO_LOW_DIM_SIM[task_name]
    if robot_state_len > 0:
        space_dict["low_dim_state"] = spaces.Box(
            -np.inf, np.inf, shape=(robot_state_len,), dtype=np.float32
        )
    for cam, name in [
        (observation_config.left_shoulder_camera, "left_shoulder"),
        (observation_config.right_shoulder_camera, "right_shoulder"),
        (observation_config.front_camera, "front"),
        (observation_config.wrist_camera, "wrist"),
        (observation_config.overhead_camera, "overhead"),
    ]:
        space_dict.update(_get_cam_observation_elements(cam, name))
    space_dict["lang_tokens"] = spaces.Box(0, 50000, shape=(1, 77), dtype=np.int32)
    return spaces.Dict(space_dict)


def _extract_obs(obs: Observation, observation_config, robot_state_keys: dict = None):
    obs_dict = vars(obs)
    desc = obs.misc["descriptions"]
    if robot_state_keys is not None:
        obs_dict = {
            k: None if k in robot_state_keys else v for k, v in obs_dict.items()
        }
        obs = Observation(**obs_dict)
    robot_state = obs.get_low_dim_data()

    obs_dict = {
        k: v for k, v in obs_dict.items() if v is not None and k not in ROBOT_STATE_KEYS
    }

    obs_dict = {
        k: v.transpose((2, 0, 1)) if v.ndim == 3 else np.expand_dims(v, 0)
        for k, v in obs_dict.items()
    }
    obs_dict["low_dim_state"] = np.array(robot_state, dtype=np.float32)
    for k, v in [(k, v) for k, v in obs_dict.items() if "point_cloud" in k]:
        obs_dict[k] = v.astype(np.float32)

    for config, name in [
        (observation_config.left_shoulder_camera, "left_shoulder"),
        (observation_config.right_shoulder_camera, "right_shoulder"),
        (observation_config.front_camera, "front"),
        (observation_config.wrist_camera, "wrist"),
        (observation_config.overhead_camera, "overhead"),
    ]:
        if config.point_cloud:
            obs_dict["%s_camera_extrinsics" % name] = obs.misc[
                "%s_camera_extrinsics" % name
            ]
            obs_dict["%s_camera_intrinsics" % name] = obs.misc[
                "%s_camera_intrinsics" % name
            ]

    tokens = clip.tokenize([desc]).numpy()
    obs_dict["lang_tokens"] = tokens

    return obs_dict


def _get_action_mode(action_mode_type: ActionModeType):
    # joint ranges
    ACT_MIN = [
        -2.8973000049591064,
        -1.7627999782562256,
        -2.8973000049591064,
        -3.0717999935150146,
        -2.8973000049591064,
        -0.017500000074505806,
        -2.8973000049591064,
        0.0,
    ]
    ACT_RANGE = [
        5.794600009918213,
        3.525599956512451,
        5.794600009918213,
        3.002000093460083,
        5.794600009918213,
        3.7699999809265137,
        5.794600009918213,
        1.0,
    ]

    match action_mode_type:
        case ActionModeType.END_EFFECTOR_POSE:

            class CustomMoveArmThenGripper(MoveArmThenGripper):
                def action_bounds(
                    self,
                ):  ## x,y,z,quat,gripper -> 8. Limited by rlbench scene workspace
                    return (
                        np.array(
                            [-0.3, -0.5, 0.6] + 3 * [-1.0] + 2 * [0.0],
                            dtype=np.float32,
                        ),
                        np.array([0.7, 0.5, 1.6] + 4 * [1.0] + [1.0], dtype=np.float32),
                    )

            action_mode = CustomMoveArmThenGripper(
                EndEffectorPoseViaPlanning(), Discrete()
            )

        # Overrides Tiger's Delta Joint Position mode with Absolute Joint Position mode
        case ActionModeType.JOINT_POSITION:

            class CustomMoveArmThenGripper(MoveArmThenGripper):
                def action_bounds(self):
                    return (
                        np.array(ACT_MIN, dtype=np.float32),
                        np.array(ACT_MIN, dtype=np.float32)
                        + np.array(ACT_RANGE, dtype=np.float32),
                    )

            action_mode = CustomMoveArmThenGripper(JointPosition(True), Discrete())

    return action_mode


def add_demo_to_replay_buffer(wrapped_env: DemoEnv, replay_buffer: ReplayBuffer):
    """Loads demos into replay buffer by passing observations through wrappers.

    CYCLING THROUGH DEMOS IS HANDLED BY WRAPPED ENV.

    Args:
        wrapped_env: the fully wrapped environment.
        replay_buffer: replay buffer to be loaded.
    """
    is_sequential = replay_buffer.sequential
    ep = []

    # Extract demonstration episode
    obs, info = wrapped_env.reset()
    fake_action = wrapped_env.action_space.sample()
    term, trunc = False, False
    while not (term or trunc):
        next_obs, rew, term, trunc, next_info = wrapped_env.step(fake_action)
        action = next_info.pop("demo_action")
        ep.append([action, obs, rew, term, trunc, info, next_info])
        obs = next_obs
        info = next_info
    final_obs, _ = obs, info

    for act, obs, rew, term, trunc, info, next_info in ep:
        obs_and_info = {k: v[-1] for k, v in obs.items()}  # remove temporal
        obs_and_info.update({"demo": info["demo"]})
        replay_buffer.add(act, rew, term, trunc, **obs_and_info)

    if not is_sequential:
        final_obs = {k: v[-1] for k, v in final_obs.items()}
        replay_buffer.add_final(**final_obs)
