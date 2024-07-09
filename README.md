# GENIMA

### [Generative Image as Action Models](https://genima-robot.github.io)   
[Mohit Shridhar*](https://mohitshridhar.com/), [Yat Long (Richie) Lo*](https://richielo.github.io/), [Stephen James](https://stepjam.github.io/)  
Dyson Robot Learning Lab  

Genima fine-tunes Stable Diffusion to draw joint-actions on RGB observations. 

![](media/teaser_v1.gif)

This repo is for reproducing the RLBench results from the paper. For the latest updates, see: [genima-robot.github.io](https://genima-robot.github.io)

*Note: This is not an official Dyson product.*

## Guides

- Getting Started: [Installation](#installation), [Quickstart](#quickstart), [Checkpoints and Pre-Generated Datasets](#download), [Model Card](model-card.md)
- Training & Evaluation: [Full Pipeline](#training-guide), [Colosseum Tests](#colosseum-perturbation-tests), [Other Diffusion Base Models](diffusion/README.md)
- Miscellaneous: [Notebooks](#notebooks), [Disclaimers](#disclaimers-and-limitations), [FAQ](#faq), [Licenses](#licenses)
- Acknowledgements: [Acknowledgements](#acknowledgements), [Citations](#citations)

## Installation

Genima is built with Python 3.10.12. We use poetry to manage dependencies.

```bash
cd <install_dir>
conda create -p genima_env python==3.10.12        # create conda env
conda activate genima_env                         # activate env

pip install poetry
poetry self add poetry-exec-plugin                # install plugin for executables
poetry self update

cd <install_dir>
git clone https://github.com/MohitShrdhar/genima.git
cd genima
poetry exec rlbench                               # install pyrep and rlbench
poetry install                                    # install dependencies
```

## Quickstart

This is a quick tutorial on evaluating a pre-trained Genima agent.  

Download the [pre-trained checkpoint]() trained on 25 RLBench tasks with 50 demos per task:
```bash
cd genima
poetry exec quick_start
```

Generate a small `val` set of 10 episodes for `open_box` inside `/tmp/val_data`:

```bash
mkdir /tmp/val_data
cd genima/rlbench/tools
python dataset_generator.py \
     --save_path=/tmp/val_data \
     --tasks=open_box \
     --image_size=256,256 \
     --renderer=opengl \
     --episodes_per_task=10 \
     --variations=1 \
     --processes=1 \
     --arm_max_velocity 2.0 \
     --arm_max_acceleration 8.0
```

Evaluate the pre-trained Genima agent:

```bash
cd genima/controller
python eval_genima.py \
     task=open_box \
     dataset_root=/tmp/val_data \
     diffusion_ckpt=../ckpts/25_tasks/diffusion_sdturbo_R256x4_tiled \
     controller_ckpt=../ckpts/25_tasks/controller_act \
     num_eval_episodes=10 \
     save_gen_images=False \
     num_diffusion_steps=5 \
     execution_horizon=20 \
     save_video=True \
     wandb.use=True \
     eval_type=latest \
     headless=False
```

If you are on a headless machine, turn off RLBench visualization with `headless=True`. You can save the generated target images to `/tmp/` by setting `save_gen_images=True`. 

You can evaluate the same Genima agent on other tasks by generating a val set for that task.  


## Download

### Pre-trained checkpoints

We provide pre-trained checkpoints for RLBench agents:

##### [25 Task Genima](https://github.com/MohitShridhar/genima/releases/download/v1.0.0/25_tasks.zip)

##### [3 Task Genima](https://github.com/MohitShridhar/genima/releases/download/v1.0.0/3_tasks.zip) - from ablations

See [quickstart](#quickstart) on how to evaluate these checkpoints. 

### RLBench datasets

See this [GDrive folder]() with `train` and `val` zips for 25 tasks. 

## Training Guide

This guide covers how to train Genima from scratch.

### 1. Generate RLBench datasets

Use the `dataset_generator.py` tool to generate datasets:
```bash
cd rlbench/tools

# generate train set
python dataset_generator.py \
     --save_path=/tmp/train_data \
     --tasks=take_lid_off_saucepan \
     --image_size=256,256 \
     --renderer=opengl \
     --episodes_per_task=25 \
     --variations=1 \
     --processes=1 \
     --arm_max_velocity 2.0 \
     --arm_max_acceleration 8.0


# generate val set
python dataset_generator.py \
     --save_path=/tmp/val_data \
     --tasks=take_lid_off_saucepan \
     --image_size=256,256 \
     --renderer=opengl \
     --episodes_per_task=10 \
     --variations=1 \
     --processes=1 \
     --arm_max_velocity 2.0 \
     --arm_max_acceleration 8.0
```

**Note:** If you have old RLBench datasets, they won't work with Genima. You need [RLBench](https://github.com/stepjam/RLBench) `master` up until [this commit](https://github.com/stepjam/RLBench/commit/4c35bc6351986a1baa6c97ab7b8fcf99395a6a17) to save joint poses.

OR  

Download pre-generated datasets from [GDrive](https://rclone.org/drive/) using rclone. 

### 2. Render joints as spheres

To draw actions in your rlbench dataset, you need to provide paths to your rlbench dataset and random textures:
```bash
# downloads textures for random backgrounds
poetry exec download_textures

# use pyrender to place spheres that at joint-actions that t+20 timesteps ahead
cd render
python render_data.py \
     episodes=25 \
     dataset_root=/tmp/train_data \
     textures_path=./mil_textures \
     action_horizon=20 \
     num_processes=5
```
By default, two dataset folders would be generated: `rlbench_data_rgb_rendered` with observations and joint targets to train the diffusion agent, and `rlbench_data_rnd_bg` with random backgrounds and joint targets to train the controller. See the sample [notebook]() for visual illustrations of the rendered data.

### 3. Fine-tune Stable Diffusion with ControlNet to draw spheres

```bash
# setup your accelerate
accelerate config

# finetune SD-turbo with controlnet
cd diffusion
python train_controlnet_genima.py
     --pretrained_model_name_or_path='stabilityai/sd-turbo' \
     --output_dir=/tmp/diffusion_agent \
     --resolution=512 \
     --learning_rate=1e-5 \
     --data_path='/tmp/train_data_rgb_rendered/' \
     --validation_images_path '/tmp/train_data_rgb_rendered' \
     --train_batch_size=2 \
     --checkpoints_total_limit=2 \
     --num_train_epochs=100 \
     --report_to wandb \
     --report_name 'sdturbo_1task_R256x4_tiled' \
     --image_type 'tiled_rgb_rendered' \
     --conditioning_image_type 'tiled_rgb' \
     --tasks 'take_lid_off_saucepan' \
     --validation_steps 500 \
     --mixed_precision='fp16' \
     --variant='fp16' \
     --allow_tf32 \
     --enable_xformers_memory_efficient_attention \
     --tiled \
     --num_validation_images 1 \
     --augmentations=crop,colorjitter \
     --num_demos 25 \
     --checkpointing_steps 1000 \
     --resume_from_checkpoint 'latest'
```

Monitor the training on wandb to check the quality of the generated targets. If the spheres are blurry, at the wrong location, or have the wrong color, then the model is not trained enough. You need train between 100-200 epochs for good results. 


### 4. Train an ACT controller to follow spheres

```bash
# train ACT to map target images to a sequence of joint-actions
cd controller
python train_act.py \
     env=rlbench \
     env.dataset_root=/tmp/train_data_rnd_bg/ \
     work_dir=/tmp/controller \
     demos=25 \
     env.tasks=[take_lid_off_saucepan] \
     num_train_epochs=1000 \
     action_sequence=20 \
     batch_size=8 \
     wandb.use=true
```

The ACT controller for Genima can be trained independently of the diffusion agent. If you have sufficient compute, you train both the diffusion agent and controller simultaneously.  

The hyperparameters of the controller are set in [`controller/cfgs/method/genima_act.yaml`](controller/cfgs/method/genima_act.yaml).

To train the ACT baseline, set `env.dataset_root=/tmp/train_data` to use raw RGB observations instead of target spheres with random backgrounds. See the TiGER repository for other baselines. 

### 5. Evaluate pre-trained Genima

```bash
# Use the diffusion agent and controller sequentially to evaluate
python eval_genima.py \
     task=take_lid_off_saucepan \
     dataset_root=/tmp/val_data \
     diffusion_ckpt=/tmp/diffusion_agent/sdturbo_1task_R256x4_tiled \
     controller_ckpt=/tmp/controller \
     num_eval_episodes=10 \
     save_gen_images=False \
     num_diffusion_steps=5 \
     execution_horizon=20 \
     save_video=True \
     wandb.use=True \
     eval_type=last_three \
     headless=True
```

To run the evaluation offline, set `headless=False`. By setting `eval_type=last_three`, the script will sequentially evaluate the last three checkpoints and report average scores. Alternatively, you can set `eval_type=latest` or `eval_type=980` for specific checkpoints.  

You can visualize the generated targets by setting `save_gen_images=True`. This will save the diffusion outputs to `/tmp`. However, note that saving images to disk is slow.  

For the fastest inference speed, set `torch_compile=True` and `enable_xformers_memory_efficient_attention=False`. See other optimizations [here](https://huggingface.co/blog/simple_sdxl_optimizations). 

All RLBench experiments in the paper use `num_diffusion_steps=10`, `execution_horizon=20`, `num_eval_episodes=50`, and `eval_type=last_three`.

## Colosseum Perturbation Tests

You can evaluate the same checkpoints from [quickstart](#quickstart) on 6 pertubation categories from [Colosseum](https://robot-colosseum.github.io/).

```bash
python eval_genima.py \
     task=open_drawer \
     dataset_root=/tmp/val_data \
     diffusion_ckpt=/tmp/diffusion_agent/sdturbo_1task_R256x4_tiled \
     controller_ckpt=/tmp/controller \
     save_gen_images=False \
     num_eval_episodes=10 \
     save_video=True \
     wandb.use=True \
     eval_type=last_three \
     headless=True \
     colosseum_use=True \
     colosseum_task_config=cfgs/colosseum/random_object_color.yaml
```

Select from 6 config files for `colosseum_task_config`:
- Randomized object and part colors: [`cfgs/colosseum/random_object_color.yaml`](controller/cfgs/colosseum/random_object_color.yaml) for `open_drawer`.
- Distractor objects: [`controller/cfgs/colosseum/distractor_objects.yaml`](controller/cfgs/colosseum/distractor_objects.yaml) for `open_drawer`.
- Lighting variations: [`controller/cfgs/colosseum/lighting_variations.yaml`](controller/cfgs/colosseum/lighting_variations.yaml) for `open_drawer`.
- Randomized background textures: [`controller/cfgs/colosseum/random_background_textures.yaml`](controller/cfgs/colosseum/random_background_textures.yaml) for `move_hanger`.
- Randomized table textures: [`controller/cfgs/colosseum/random_table_textures.yaml`](controller/cfgs/colosseum/random_table_textures.yaml) for `basketball_in_hoop`.
- Camera pose perturbations: [`controller/cfgs/colosseum/random_camera_poses.yaml`](controller/cfgs/colosseum/random_camera_poses.yaml) for `move_hanger`.

## Notebooks

- [Colab Tutorial](): How to generate the dataset and train the diffusion agent.
- [Dataset Visualizer](notebooks/render.ipynb): Looks into the rendered joint-target datasets.  

## Disclaimers and Limitations

- **Parallelization:** A lot of things (data generation, evaluation) are slow because everything is done serially. Parallelizing these processes will save you a lot of time.
- **Variance in Success Rates:** You may notice small variations in the success rate due to the stochastic nature of the simulator. 
- **Other Limitations:** See the "Limitations and Potential Solutions" section in the paper appendix.

## FAQ

**How long should I train for?**

100-200 epochs for the diffusion agent. 1000 epochs for the controller. 

**How many training demos do I need?**

It depends on the number, complexity, and diversity of tasks. Start with 50 demos in simulation and iteratively reduce the number demos until you achieve >80% of the peak performance. 

**Is multi-gpu training supported?**  

Yes for the diffusion agent, since it's based off [HF diffusers](https://huggingface.co/docs/diffusers/en/index). But no for the controller, since TiGER only supports single-GPU training. You can use other ACT implementations to train the controller. 

**Will the real-robot code be released?**

The Genima part of the real-robot code is identical to this repo. You just need format your dataset into the RLBench dataset format. 

## Hardware Requirements


- Diffusion Agent Training: A100 with 80GB VRAM
- Controller Training: L4 with 24GB VRAM, 120GB RAM
- Evaluation: L4 or RTX 3090 with 24GB VRAM

Only the diffusion agent training requires GPUs with larger VRAMs. Both inference and controller training can be done on commodity GPUs.

## Release Notes

**Update 28-Aug-2024:**

- Initial code release. 

## Licenses
- [Genima License (Apache 2.0)](LICENSE) - This repo
- [Huggingface Diffusers License (Apache 2.0)](https://github.com/huggingface/diffusers?tab=Apache-2.0-1-ov-file#readme) - Fine-tuning stable diffusion code
- [TiGER License (MIT)](https://github.com/dyson-ai/tiger/blob/main/LICENSE) - ACT implementation and training code
- [ACT License (MIT)](https://github.com/tonyzhaozh/act?tab=MIT-1-ov-file#readme) - Original ACT code
- [MT-ACT License (MIT)](https://github.com/robopen/roboagent/) - Original multi-task ACT code
- [RLBench Licence](https://github.com/stepjam/RLBench/blob/master/LICENSE) - Simulator
- [PyRep License (MIT)](https://github.com/stepjam/PyRep/blob/master/LICENSE) - Simulator wrapper
- [CLIP License (MIT)](https://github.com/openai/CLIP/blob/main/LICENSE) - Language encoder


## Acknowledgements

Special thanks to Huggingface for [Diffusers](https://github.com/huggingface/diffusers), Zhao et al. for the [ACT repo](https://github.com/tonyzhaozh/act), and Bharadhwaj et al. for the [MT-ACT repo](https://github.com/robopen/roboagent/). 

## Citations

**Genima**
```
@inproceedings{shridhar2024genima,
  title     = {Generative Image as Action Models},
  author    = {Shridhar, Mohit and Lo, Yat Long and James, Stepheb},
  booktitle = {Proceedings of the 6th Conference on Robot Learning (CoRL)},
  year      = {2024},
}
```

**Diffusers**
```
@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Dhruv Nair and Sayak Paul and William Berman and Yiyi Xu and Steven Liu and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```

**ACT**
```
@inproceedings{zhao2023learning,
  title={Learning fine-grained bimanual manipulation with low-cost hardware},
  author={Zhao, Tony Z and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  booktitle = {Robotics: Science and Systems (RSS)},
  year={2023}
}
```

**MT-ACT**
```
@misc{bharadhwaj2023roboagent,
  title={RoboAgent: Generalization and Efficiency in Robot Manipulation via Semantic Augmentations and Action Chunking},
  author={Homanga Bharadhwaj and Jay Vakil and Mohit Sharma and Abhinav Gupta and Shubham Tulsiani and Vikash Kumar},
  year={2023},
  eprint={2309.01918},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```

**Colosseum**
```
@inproceedings{pumacay2024colosseum,
  title     = {THE COLOSSEUM: A Benchmark for Evaluating Generalization for Robotic Manipulation},
  author    = {Pumacay, Wilbert and Singh, Ishika and Duan, Jiafei and Krishna, Ranjay and Thomason, Jesse and Fox, Dieter},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2024},
}
```