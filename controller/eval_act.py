import os
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

import json
import time
import torch
import numpy as np

import tiger.utils as tiger_utils
from tiger.workspace import Workspace
from tiger.envs.env import EnvFactory
from tiger.logger import Logger
from env.rlbench import GenimaRLBenchFactory


class GenimaEvalWorkspace(Workspace):
    def __init__(
        self,
        train_cfg: DictConfig,
        eval_cfg: DictConfig,
        env_factory: EnvFactory,
        work_dir: str = None,
    ):
        assert env_factory is not None

        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg

        self.work_dir = Path(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            if work_dir is None
            else work_dir
        )
        print(f"workspace: {self.work_dir}")

        tiger_utils.set_seed_everywhere(self.train_cfg.seed)
        self.device = torch.device(self.eval_cfg.device)

        # create logger
        self.logger = Logger(self.work_dir, cfg=self.train_cfg)
        self.env_factory = env_factory
        self.eval_env = self.env_factory.make_eval_env(
            self.train_cfg, self.eval_cfg.controller_ckpt
        )

        # Create the controller agent
        observation_space = self.eval_env.observation_space
        action_space = self.eval_env.action_space

        self.controller_agent = hydra.utils.instantiate(
            self.train_cfg.method,
            device=self.device,
            observation_space=observation_space,
            action_space=action_space,
            num_train_envs=self.train_cfg.num_train_envs,
            replay_alpha=self.train_cfg.replay.alpha,
            replay_beta=self.train_cfg.replay.beta,
            frame_stack_on_channel=self.train_cfg.frame_stack_on_channel,
        )
        self.controller_agent.train(False)

        # RLBench doesn't like it when we import cv2 before it, so moving
        # import here.
        from tiger.video import VideoRecorder

        self.eval_video_recorder = VideoRecorder(
            Path(os.path.join(self.eval_cfg.controller_ckpt, "eval_videos"))
            if self.eval_cfg.save_video
            else None
        )

        self._timer = tiger_utils.Timer()
        self._pretrain_step = 0
        self._main_loop_iterations = 0
        self._global_env_episode = 0
        self._act_dim = self.eval_env.action_space.shape[0]
        self._episode_rollouts = []
        self._shutting_down = False

    def load_controller_ckpt(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.eval_cfg.device)

        missing_keys = [
            k
            for k in self.controller_agent.state_dict().keys()
            if k not in checkpoint["agent"].keys() and "clip" not in k
        ]
        if len(missing_keys) > 0:
            raise ValueError(f"Missing keys in controller checkpoint: {missing_keys}")

        self.controller_agent.load_state_dict(checkpoint["agent"], strict=False)
        print(f"Loaded controller checkpoint from {checkpoint_path}")

    def eval_checkpoints(self, eval_ckpts):
        print(f"Checkpoints to evaluate: {eval_ckpts}")

        # Global log placeholders
        logs = {"eval_episodes": []}
        logs_name = f"eval_genima_{self.eval_cfg.task}.json"
        logs_path = os.path.join(self.eval_cfg.controller_ckpt, logs_name)

        # Sequentially evaluate the checkpoints
        global_episode, global_total_reward = 0, 0
        for run_id, eval_ckpt in enumerate(eval_ckpts):
            print(f"\n-------- Run {run_id} ---------")

            # Load the controller checkpoint
            controller_ckpt_path = os.path.join(
                self.eval_cfg.controller_ckpt, eval_ckpt
            )
            self.load_controller_ckpt(controller_ckpt_path)

            run_episode, run_total_reward = 0, 0
            timings = {"gen_time": [], "control_time": []}
            eval_until_episode = tiger_utils.Until(self.eval_cfg.num_eval_episodes)

            while eval_until_episode(run_episode):
                # Reset env
                obs, info = self.eval_env.reset()

                # Restore initial state in evaluation episode
                _, obs = self.eval_env.reset_to_demo(idx=run_episode)

                # Episode log placeholders
                termination, episode_step = False, 0

                # Use waypoint0 from RLBench as the initial object pose
                # (used for visualization purposes only)
                initial_object_pose = (
                    self.eval_env.unwrapped._rlbench_env._scene._workspace.get_object(
                        "waypoint0"
                    ).get_pose()
                )

                # Initialize video recorder
                self.eval_video_recorder.init(
                    self.eval_env,
                    enabled=self.eval_cfg.save_video,
                )

                # Evaluation loop
                while not termination:

                    with torch.inference_mode(), tiger_utils.eval_mode(
                        self.controller_agent
                    ):
                        # Generate a sequence of joint-positions with the controller
                        obs = {
                            k: torch.from_numpy(v).to(self.eval_cfg.device).unsqueeze(0)
                            for k, v in obs.items()
                        }

                        control_start_time = time.time()
                        actions = self.controller_agent.act(
                            obs,
                            step=episode_step,
                            eval_mode=True,
                        )[0]
                        actions = actions.detach().cpu().numpy()
                        control_time = time.time() - control_start_time

                        # Step actions with environment
                        try:
                            obs, reward, termination, _, info = self.eval_env.step(
                                actions
                            )
                        except Exception as e:
                            print(f"Error: {e}")
                            termination = True
                            break

                        episode_step += len(
                            actions
                        )  # TODO: switch to execution horizon

                        # Save timings
                        timings["control_time"].append(control_time)

                        # Save video frame
                        self.eval_video_recorder.record(self.eval_env)

                        # Timeout by exceeding episode length
                        if episode_step > self.train_cfg.env.episode_length:
                            termination = True
                            break

                run_total_reward += float(reward)
                run_episode += 1
                global_total_reward += float(reward)
                global_episode += 1

                # Save episode metrics to json file
                logs["eval_episodes"].append(
                    {
                        "episode": run_episode,
                        "reward": float(reward),
                        "global_episode": global_episode,
                        "global_reward": global_total_reward,
                        "steps": episode_step,
                        "run_id": run_id,
                        "controller_ckpt": eval_ckpt,
                        "initial_object_pose": list(initial_object_pose),
                    }
                )

                with open(logs_path, "w") as f:
                    json.dump(logs, f, indent=4)

                # Log to wandb
                metrics = {
                    "reward": float(reward),
                    "success": global_total_reward / float(global_episode),
                    "episode": global_episode,
                    "control_time": np.mean(timings["control_time"]),
                }

                # Save video to wandb, if enabled
                if self.eval_cfg.save_video:
                    metrics["eval_rollout"] = dict(
                        video=np.array(self.eval_video_recorder.frames), fps=4
                    )

                    success = "succ" if reward > 0.9 else "fail"
                    self.eval_video_recorder.save(
                        f"{self.eval_cfg.task}_ep{global_episode}_{success}.mp4"
                    )

                self.logger.log_metrics(metrics, global_episode, prefix="eval_act")

                # Print episode metrics
                print(
                    (
                        f"Episode {run_episode:>02}\t| Reward - run{run_id}: {reward:.1f}"
                        f"\t({int(run_total_reward)}/{run_episode}={run_total_reward/run_episode*100:.1f}%)"
                        f"\t| Steps: {episode_step}"
                        f"\t| Control Time: {np.mean(timings['control_time']):.4f}s"
                    )
                )

        # Save global metrics to json file
        logs["results"] = {
            "avg_success": f"{global_total_reward / float(global_episode)}",
            "total_success": global_total_reward,
            "total_episodes": global_episode,
            "eval_type": self.eval_cfg.eval_type,
        }
        with open(logs_path, "w") as f:
            json.dump(logs, f, indent=4)

        print("----------------------")
        print(
            f"Average of {run_episode} episodes (across {len(eval_ckpts)} runs): "
            f"{global_total_reward / float(global_episode)*100:.2f}%"
        )

    def eval(self):
        # Check if controller checkpoint directory exists
        controller_ckpt = Path(self.eval_cfg.controller_ckpt)
        if not controller_ckpt.exists():
            raise ValueError(f"Controller checkpoint not found at {controller_ckpt}")

        # Choose the relevant checkpoints
        ckpts = [
            f
            for f in os.listdir(controller_ckpt)
            if f.endswith(".pt") and f != "latest.pt"
        ]
        ckpts.sort(key=lambda x: int(x.split(".")[0]))
        ckpt_steps = [int(f.split(".")[0]) for f in ckpts]

        if self.eval_cfg.eval_type == "latest":
            eval_ckpts = ["latest.pt"]
        elif self.eval_cfg.eval_type == "last_three":
            eval_ckpts = [
                "latest.pt",
                ckpts[-2],
                ckpts[-3],
            ]  # NOTE: not exactly last three
        elif self.eval_cfg.eval_type == "last":
            eval_ckpts = [ckpts[-1]]
        elif int(self.eval_cfg.eval_type) in ckpt_steps:
            eval_ckpts = [str(self.eval_cfg.eval_type) + ".pt"]
        else:
            raise ValueError(f"Invalid eval_type {self.eval_cfg.eval_type}")

        # Reset environment
        self.eval_env.reset()

        # Evaluate the checkpoints
        self.eval_checkpoints(eval_ckpts)

        # Close envs
        self.eval_env.close()


@hydra.main(config_path="cfgs", config_name="eval_act", version_base=None)
def main(eval_cfg):
    train_cfg_path = Path(eval_cfg.train_cfg_path)
    train_cfg = OmegaConf.load(train_cfg_path)

    # override train env configs for evaluation
    # this train cfg is used to initialize the eval env and agent
    train_cfg.env.headless = eval_cfg.headless
    train_cfg.env.task_name = eval_cfg.task
    train_cfg.env.dataset_root = eval_cfg.dataset_root
    train_cfg.env.episode_length = eval_cfg.episode_length
    train_cfg.env.colosseum_use = eval_cfg.colosseum_use
    train_cfg.env.colosseum_task_config = eval_cfg.colosseum_task_config
    train_cfg.wandb = eval_cfg.wandb

    print(f"Evaluating on {train_cfg.env.task_name} task")

    workspace = GenimaEvalWorkspace(
        train_cfg, eval_cfg, env_factory=GenimaRLBenchFactory()
    )

    workspace.eval()


if __name__ == "__main__":
    main()
