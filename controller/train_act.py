"""Launch file for running experiments in RoboBase."""

import os
import sys
import logging
import traceback

from pathlib import Path
from typing import Callable
import copy
import tqdm
from functools import partial

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from robobase import utils
from robobase.logger import Logger
from robobase.envs.env import EnvFactory
from robobase.replay_buffer.replay_buffer import ReplayBuffer
from robobase.workspace import Workspace, _worker_init_fn
from env.rlbench import GenimaRLBenchFactory
from utils.dataloader import EpochReplayBuffer
from torch.utils.data import DataLoader
import torch
import numpy as np
from gymnasium import spaces
import gymnasium as gym
from natsort import natsorted

log = logging.getLogger(__name__)


def _create_default_replay_buffer(
    cfg: DictConfig,
    observation_space: gym.Space,
    action_space: gym.Space,
    demo_replay: bool = False,
) -> ReplayBuffer:
    extra_replay_elements = spaces.Dict({})
    if cfg.demos > 0:
        extra_replay_elements["demo"] = spaces.Box(0, 1, shape=(), dtype=np.uint8)
    # Create replay_class with buffer-specific hyperparameters
    replay_class = EpochReplayBuffer
    replay_class = partial(
        replay_class,
        nstep=cfg.replay.nstep,
        gamma=cfg.replay.gamma,
    )
    # Create replay_class with common hyperparameters
    return replay_class(
        save_dir=cfg.replay.save_dir,
        batch_size=cfg.batch_size if not demo_replay else cfg.demo_batch_size,
        replay_capacity=cfg.replay.size if not demo_replay else cfg.replay.demo_size,
        action_shape=action_space.shape,
        action_dtype=action_space.dtype,
        reward_shape=(),
        reward_dtype=np.float32,
        observation_elements=observation_space,
        extra_replay_elements=extra_replay_elements,
        # num_workers=cfg.replay.num_workers,
        num_workers=0,
        sequential=cfg.replay.sequential,
    )


class ControllerWorkspace(Workspace):
    def __init__(
        self,
        cfg: DictConfig,
        env_factory: EnvFactory = None,
        create_replay_fn: Callable[[DictConfig], ReplayBuffer] = None,
        work_dir: str = None,
    ):
        assert env_factory is not None

        if create_replay_fn is None:
            create_replay_fn = _create_default_replay_buffer

        self.exp_path_str = os.path.join(work_dir, "snapshots", cfg.experiment_name)
        self.work_dir = Path(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            if work_dir is None
            else work_dir
        )
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        dev = "cpu"
        if cfg.num_gpus > 0:
            if sys.platform == "darwin":
                dev = "mps"
            else:
                dev = 0
                job_num = False
                try:
                    job_num = HydraConfig.get().job.get("num", False)
                except ValueError:
                    pass
                if job_num:
                    dev = job_num % cfg.num_gpus
        self.device = torch.device(dev)
        # create logger
        self.logger = Logger(self.work_dir, cfg=self.cfg)
        self.env_factory = env_factory

        assert cfg.demos > 0, print("Demonstrations are needed to train controller.")
        # Collect demos or fetch saved demos before making environments
        # to consider demo-based action space (e.g., standardization)
        self.env_factory.collect_or_fetch_demos(cfg, cfg.demos)

        # Compute action statistics from demostrations

        # Create evaluation environment
        # This is just to retrieve the observation and action spaces
        temp_env_config = copy.deepcopy(cfg)
        OmegaConf.set_struct(temp_env_config, True)
        with open_dict(temp_env_config):
            temp_env_config.env.episode_length = 125
            temp_env_config.env.task_name = "take_lid_off_saucepan"
        eval_env = self.env_factory.make_eval_env(
            temp_env_config,
            stats_path=self.exp_path_str,
            action_stats=self.env_factory._action_stats,
            proprio_stats=self.env_factory._proprio_stats,
        )

        # Post-process demos using the information from environments
        self.env_factory.post_collect_or_fetch_demos(cfg)

        # Create the Controller Agent
        observation_space = eval_env.observation_space
        action_space = eval_env.action_space
        intrinsic_reward_module = None
        self.agent = hydra.utils.instantiate(
            cfg.method,
            device=self.device,
            observation_space=observation_space,
            action_space=action_space,
            num_train_envs=cfg.num_train_envs,
            replay_alpha=cfg.replay.alpha,
            replay_beta=cfg.replay.beta,
            frame_stack_on_channel=cfg.frame_stack_on_channel,
            intrinsic_reward_module=intrinsic_reward_module,
        )
        self.agent.train(False)
        self.use_demo_replay = False
        self.replay_buffer = create_replay_fn(cfg, observation_space, action_space)
        self.prioritized_replay = cfg.replay.prioritization
        self.extra_replay_elements = self.replay_buffer.extra_replay_elements

        self.replay_loader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.batch_size,
            num_workers=0,
            pin_memory=cfg.replay.pin_memory,
            worker_init_fn=_worker_init_fn,
        )
        self._replay_iter = None

        if self.prioritized_replay:
            if self.use_demo_replay:
                raise NotImplementedError(
                    "Demo replay is not compatible with prioritized replay"
                )

        # RLBench doesn't like it when we import cv2 before it, so moving
        # import here.
        from robobase.video import VideoRecorder

        self.eval_video_recorder = VideoRecorder(
            (self.work_dir / "eval_videos") if self.cfg.log_eval_video else None
        )

        self._timer = utils.Timer()
        self._pretrain_step = 0
        self._main_loop_iterations = 0
        self._global_env_episode = 0
        self._act_dim = eval_env.action_space.shape[0]
        self._episode_rollouts = []
        self._epoch = 0
        self._num_iters = 0

        eval_env.close()
        eval_env = None

        self._shutting_down = False

    def _perform_updates(self):
        metrics = {}
        metrics.update(self.agent.update(self.replay_iter, 0, self.replay_buffer))
        return metrics

    def _train(self):
        ckpt_dir = os.path.join(self.work_dir, "snapshots", self.cfg.experiment_name)
        snapshot_path = os.path.join(ckpt_dir, "latest.pt")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        elif os.path.isfile(snapshot_path):
            # load latest checkpoint
            self.load_snapshot(snapshot_path)
            print(f"Snapshot loaded from : {snapshot_path}")

        self.agent.train(True)
        self.agent.logging = True
        for epoch in tqdm.tqdm(
            range(self._epoch, self.cfg.num_train_epochs),
            desc="Epoch",
            initial=self._epoch,
            total=self.cfg.num_train_epochs,
        ):
            replay_iter = iter(self.replay_buffer)
            while True:
                try:
                    metrics = {}
                    metrics.update(
                        self.agent.update(
                            replay_iter, self._num_iters, self.replay_buffer
                        )
                    )
                    self.logger.log_metrics(
                        metrics, self._num_iters, "train_controller"
                    )
                    self._num_iters += 1
                except StopIteration:
                    break
                except Exception as e:
                    print(f"Unexpected error in training: {e}")
                    logging.error(traceback.format_exc())

            # Checkpointing
            if epoch % self.cfg.checkpoint_every == 0:
                # Rename old checkpoint
                if os.path.exists(os.path.join(ckpt_dir, "latest.pt")):
                    os.rename(
                        os.path.join(ckpt_dir, "latest.pt"),
                        os.path.join(
                            ckpt_dir, f"{max(0, epoch-self.cfg.checkpoint_every)}.pt"
                        ),
                    )

                ckpts = natsorted(
                    [
                        pt
                        for pt in os.listdir(ckpt_dir)
                        if pt.endswith(".pt") and pt != "latest.pt"
                    ]
                )
                if len(ckpts) > self.cfg.num_checkpoints:
                    for i in range(len(ckpts) - self.cfg.num_checkpoints):
                        os.remove(os.path.join(ckpt_dir, ckpts[i]))

                self.save_snapshot("latest", epoch)

        self.agent.train(False)

    def train(self):
        self._load_demos()
        self._train()

    def save_snapshot(self, ckpt_name, step):
        experiment_dir = self.work_dir / "snapshots" / self.cfg.experiment_name
        snapshot = experiment_dir / f"{ckpt_name}.pt"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        keys_to_save = ["cfg"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload["_epoch"] = step
        payload["_num_iters"] = self._num_iters
        state_dict = self.agent.state_dict()
        # Filter out clip model keys
        state_dict = {k: v for k, v in state_dict.items() if "clip_model" not in k}
        payload["agent"] = state_dict
        with snapshot.open("wb") as f:
            torch.save(payload, f)

        config_path = experiment_dir / "config.yaml"
        if not config_path.exists():
            OmegaConf.save(self.cfg, config_path)


@hydra.main(config_path="cfgs", config_name="controller", version_base=None)
def main(cfg):
    """Main.

    Args:
        cfg: Hydra config.
    """
    workspace = ControllerWorkspace(
        cfg, env_factory=GenimaRLBenchFactory(), work_dir=cfg.work_dir
    )
    workspace.train()


if __name__ == "__main__":
    main()
