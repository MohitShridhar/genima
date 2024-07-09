import copy
from typing import List
import warnings
from pyrep.const import RenderMode

from tiger.envs.rlbench import RLBenchEnvFactory, RLBenchEnv, ActionModeType
from tiger.envs.rlbench import ROBOT_STATE_KEYS, _make_obs_config
from tiger.envs.wrappers import FrameStack, ActionSequence, AppendDemoInfo, OnehotTime
from tiger.envs.env import Demo, DemoEnv
from tiger.utils import (
    observations_to_action_with_onehot_gripper_nbp,
    observations_to_timesteps,
)
from tiger.utils import DemoStep, rescale_demo_actions
from tiger.envs.rlbench import _name_to_task_class
from env.wrappers.proprio_norm_wrapper import NormProprioFromStats
from env.wrappers.action_normalization_wrapper import JointNormalization
from gymnasium.wrappers import TimeLimit
from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import ActionMode
from omegaconf import DictConfig, OmegaConf, open_dict
import time
import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces
import multiprocessing as mp
from rlbench import Environment

from env.rlbench_utils import _convert_rlbench_demos_for_loading
from env.rlbench_utils import observations_to_action_with_onehot_gripper
from env.rlbench_utils import _observation_config_to_gym_space
from env.rlbench_utils import _extract_obs, _get_action_mode, add_demo_to_replay_buffer


def _update_task_name_config(cfg, task_name):
    new_env_config = copy.deepcopy(cfg)
    OmegaConf.set_struct(new_env_config, True)
    with open_dict(new_env_config):
        new_env_config.env.task_name = task_name
    return new_env_config


def _get_demo_fn(cfg, num_demos, demo_list):
    obs_config = _make_obs_config(cfg)
    obs_config_demo = copy.deepcopy(obs_config)

    # RLBench demos are all saved in same action mode (joint).
    # For conversion to an alternate action mode, additional
    # info may be required. ROBOT_STATE_KEYS is altered to
    # reflect this and ensure low_dim_state is consitent
    # for demo and rollout steps.

    match ActionModeType[cfg.env.action_mode]:
        case ActionModeType.END_EFFECTOR_POSE:
            obs_config_demo.joint_velocities = True
            obs_config_demo.gripper_matrix = True

        case ActionModeType.JOINT_POSITION:
            pass

        case _:
            raise ValueError(f"Unsupported action mode type: {cfg.env.action_mode}")

    # Get common true attribute in both configs and alter ROBOT_STATE_KEYS
    common_true = [
        attr_name
        for attr_name in dir(obs_config_demo)
        if isinstance(getattr(obs_config_demo, attr_name), bool)
        and getattr(obs_config_demo, attr_name)
        and getattr(obs_config, attr_name)
        # if "camera" not in attr_name
    ]
    demo_state_keys = copy.deepcopy(ROBOT_STATE_KEYS)
    for attr in common_true:
        demo_state_keys.remove(attr)

    for task_name in cfg.env.tasks:
        updated_env_config = _update_task_name_config(cfg, task_name)
        rlb_env = _make_env(updated_env_config, obs_config_demo)
        _, info = rlb_env.reset(robot_state_keys=demo_state_keys)
        desc = info["descriptions"]
        demos = rlb_env.get_demos(num_demos, desc, robot_state_keys=demo_state_keys)
        demo_list.extend(demos)
        print(f"Loaded demos for {task_name}, total num of demos: {len(demo_list)}")
        rlb_env.close()


def _make_env(cfg: DictConfig, obs_config: dict):
    # NOTE: Completely random initialization
    # TODO: Can we make this deterministic based on cfg.seed?
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    task_name = cfg.env.task_name
    action_mode = _get_action_mode(ActionModeType[cfg.env.action_mode])

    return GenimaRLBenchEnv(
        task_name,
        obs_config,
        action_mode,
        action_mode_type=ActionModeType[cfg.env.action_mode],
        dataset_root=cfg.env.dataset_root,
        render_mode="rgb_array",
        headless=cfg.env.headless,
        colosseum_use=cfg.env.colosseum_use,
        colosseum_task_config=cfg.env.colosseum_task_config,
    )


class GenimaRLBenchEnv(RLBenchEnv):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        task_name: str,
        observation_config: ObservationConfig,
        action_mode: ActionMode,
        action_mode_type: ActionModeType = ActionModeType.JOINT_POSITION,
        dataset_root: str = "",
        renderer: str = "opengl",
        headless: bool = True,
        render_mode: str = None,
        colosseum_use: bool = False,
        colosseum_task_config: str = "cfgs/colosseum/random_object_color.yaml",
    ):
        self._task_name = task_name
        self._observation_config = observation_config
        self._action_mode = action_mode
        self._action_mode_type = action_mode_type
        self._dataset_root = dataset_root
        self._headless = headless
        self._rlbench_env = None
        self._colosseum_use = colosseum_use
        self._colosseum_task_config = colosseum_task_config
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.observation_space = _observation_config_to_gym_space(
            observation_config, task_name
        )
        minimum, maximum = action_mode.action_bounds()
        self.action_space = spaces.Box(
            minimum, maximum, shape=maximum.shape, dtype=maximum.dtype
        )
        if renderer == "opengl":
            self.renderer = RenderMode.OPENGL
        elif renderer == "opengl3":
            self.renderer = RenderMode.OPENGL3
        else:
            raise ValueError(self.renderer)

    def _launch(self):
        task_class = _name_to_task_class(self._task_name)

        if self._colosseum_use:
            from colosseum import TASKS_TTM_FOLDER
            from colosseum.rlbench.extensions.environment import EnvironmentExt

            task_cfg = OmegaConf.load(self._colosseum_task_config)
            self._rlbench_env = EnvironmentExt(
                action_mode=self._action_mode,
                obs_config=self._observation_config,
                dataset_root=self._dataset_root,
                headless=self._headless,
                path_task_ttms=TASKS_TTM_FOLDER,
                env_config=task_cfg.env,
            )
        else:
            self._rlbench_env = Environment(
                action_mode=self._action_mode,
                obs_config=self._observation_config,
                dataset_root=self._dataset_root,
                headless=self._headless,
            )

        self._rlbench_env.launch()
        self._task = self._rlbench_env.get_task(task_class)
        if self.render_mode is not None:
            self._add_video_camera()

    def get_demos(
        self, num_demos: int, desc: str, robot_state_keys: dict = None
    ) -> List[Demo]:
        live_demos = not self._dataset_root
        if live_demos:
            warnings.warn(
                "dataset_root was not defined. Generating live demos. "
                "This may take a while..."
            )
        raw_demos = self._task.get_demos(num_demos, live_demos=live_demos)
        match self._action_mode_type:
            case ActionModeType.END_EFFECTOR_POSE:
                raw_demos = self.get_nbp_demos(raw_demos)
                action_func = observations_to_action_with_onehot_gripper_nbp
            case ActionModeType.JOINT_POSITION:
                action_func = observations_to_action_with_onehot_gripper

                # NOTE: Check there is a misc["joint_position_action"]
                is_joint_position_action_included = False
                for obs in raw_demos[0]:
                    if "joint_position_action" in obs.misc:
                        is_joint_position_action_included = True
                        break
                assert is_joint_position_action_included, (
                    "`joint_position_action` is not in obs.misc, "
                    "which could severely affect performance. Please use the "
                    "latest version of PyRep and RLBench for collecting demos."
                )

        demos_to_load = _convert_rlbench_demos_for_loading(
            raw_demos,
            self._observation_config,
            desc,
            robot_state_keys=robot_state_keys,
        )

        # Process the demos using the selected action function
        loaded_demos = []
        for demo in demos_to_load:
            loaded_demos += observations_to_timesteps(
                demo, self.action_space, skipping=False, obs_to_act_func=action_func
            )
        return loaded_demos

    def reset(self, seed=None, options=None, robot_state_keys: dict = None):
        super().reset(seed=seed)
        if self._rlbench_env is None:
            self._launch()
        descs, rlb_obs = self._task.reset()
        desc = descs[0]
        rlb_obs.misc["descriptions"] = desc

        obs = _extract_obs(rlb_obs, self._observation_config, robot_state_keys)
        return obs, {"demo": 0, "descriptions": desc}

    def reset_to_demo(self, idx: int):
        assert self._dataset_root

        self._task.set_variation(0)
        (d,) = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=idx
        )

        descs, rlb_obs = self._task.reset_to_demo(d)
        desc = descs[0]
        rlb_obs.misc["descriptions"] = desc
        obs = _extract_obs(rlb_obs, self._observation_config)

        # unsqueeze each numpy array inside obs to have batch dim of 1
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                obs[k] = np.expand_dims(v, axis=0)

        return descs, obs


class GenimaRLBenchFactory(RLBenchEnvFactory):
    def make_train_env(self, cfg: DictConfig) -> gym.vector.VectorEnv:
        obs_config = _make_obs_config(cfg)

        return gym.vector.AsyncVectorEnv(
            [
                lambda: self._wrap_env(_make_env(cfg, obs_config), cfg)
                for _ in range(cfg.num_train_envs)
            ]
        )

    def make_eval_env(
        self,
        cfg: DictConfig,
        stats_path: str = "./",
        action_stats=None,
        proprio_stats=None,
    ) -> gym.Env:
        obs_config = _make_obs_config(cfg)
        self._action_stats = action_stats
        self._proprio_stats = proprio_stats
        self._action_stats_path = stats_path
        self._proprio_stats_path = stats_path
        # NOTE: Assumes workspace always creates eval_env in the main thread
        env, (self._action_space, self._observation_space) = self._wrap_env(
            _make_env(cfg, obs_config), cfg, return_raw_spaces=True
        )
        return env

    def _wrap_env(self, env, cfg, return_raw_spaces=False):
        if return_raw_spaces:
            action_space = copy.deepcopy(env.action_space)
            observation_space = copy.deepcopy(env.observation_space)
        env = JointNormalization(env, self._action_stats, self._action_stats_path)
        env = NormProprioFromStats(env, self._proprio_stats, self._proprio_stats_path)
        env = TimeLimit(env, cfg.env.episode_length)
        if cfg.use_onehot_time_and_no_bootstrap:
            env = OnehotTime(env, cfg.env.episode_length)
        env = FrameStack(env, cfg.frame_stack)
        env = ActionSequence(env, cfg.action_sequence)
        env = AppendDemoInfo(env)
        if return_raw_spaces:
            return env, (action_space, observation_space)
        else:
            return env

    def collect_or_fetch_demos(self, cfg: DictConfig, num_demos: int):
        """See base class for documentation."""

        manager = mp.Manager()
        demo_mp_list = manager.list()
        p = mp.Process(
            target=_get_demo_fn,
            args=(
                cfg,
                num_demos,
                demo_mp_list,
            ),
        )
        p.start()
        p.join()

        self._raw_demos = list(demo_mp_list)
        # Compute action statistics for demo-based rescaling, e.g., standardization
        self._action_stats = self._compute_action_stats(self._raw_demos)
        self._proprio_stats = self._compute_proprio_stats(self._raw_demos)

    def post_collect_or_fetch_demos(self, cfg: DictConfig):
        self._demos = rescale_demo_actions(
            self._rescale_demo_action_helper, self._raw_demos, cfg
        )

    def _rescale_demo_action_helper(self, info, cfg: DictConfig):
        match ActionModeType[cfg.env.action_mode]:
            case ActionModeType.END_EFFECTOR_POSE:
                raise NotImplementedError("EE pose is not supported")
            case ActionModeType.JOINT_POSITION:
                return JointNormalization.transform_to_norm(
                    info["demo_action"],
                    self._action_stats["mean"],
                    self._action_stats["std"],
                )

    def load_demos_into_replay(self, cfg: DictConfig, buffer):
        """See base class for documentation."""
        assert hasattr(self, "_demos"), (
            "There's no _demo attribute inside the factory, "
            "Check `collect_or_fetch_demos` is called before calling this method."
        )
        demo_env = self._wrap_env(
            DemoEnv(
                copy.deepcopy(self._demos), self._action_space, self._observation_space
            ),
            cfg,
        )
        for _ in range(len(self._demos)):
            add_demo_to_replay_buffer(demo_env, buffer)

    def _compute_proprio_stats(self, demos: List[List[DemoStep]]):
        """Compute statistics from demonstration actions, which could be useful for
        users that want to set action space based on demo action statistics.

        Args:
            demos: list of demo episodes

        Returns:
            Dict[str, np.ndarray]: a dictionary of numpy arrays that contain action
            statistics (i.e., mean, std, max, and min)
        """
        actions = []
        for demo in demos:
            for step in demo:
                *_, info = step
                if "demo_action" in info:
                    actions.append(info["demo_action"])
        actions = np.stack(actions)

        # Gripper one-hot action's stats are hard-coded
        action_mean = np.hstack([1 / 2, np.mean(actions, 0)[:-1]])
        action_std = np.hstack([1 / 6, np.std(actions, 0)[:-1]])
        action_max = np.hstack([1, np.max(actions, 0)[:-1]])
        action_min = np.hstack([0, np.min(actions, 0)[:-1]])
        action_stats = {
            "mean": action_mean,
            "std": action_std,
            "max": action_max,
            "min": action_min,
        }
        return action_stats
