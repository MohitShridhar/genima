# Diffusion Agent

The diffusion agent is agnostic to the choice of the fine-tuning pipeline.  

#### ControlNet with SD Turbo

```bash
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


#### ControlNet with SDXL Turbo

```bash
cd diffusion
python train_controlnet_sdxl_genima.py
     --pretrained_model_name_or_path='stabilityai/sdxl-turbo' \
     --output_dir=/tmp/diffusion_agent \
     --resolution=512 \
     --learning_rate=1e-5 \
     --data_path='/tmp/train_data_rgb_rendered/' \
     --validation_images_path '/tmp/train_data_rgb_rendered/'       \
     --validation_prompt 'a robot manipulator taking the lid off a saucepan from wrist perspective'   \
     --train_batch_size=2       \
     --checkpoints_total_limit=2       \
     --num_train_epochs=100       \
     --report_to wandb       \
     --report_name 'sdxlturbo_saucepan'       \
     --image_type 'tiled_rgb_rendered'       \
     --conditioning_image_type 'tiled_rgb'       \
     --tasks 'take_lid_off_saucepan'       \
     --validation_steps 500       \
     --mixed_precision='fp16'       \
     --variant='fp16'       \
     --allow_tf32       \
     --enable_xformers_memory_efficient_attention       \
     --tiled       \
     --num_validation_images 4       \
     --augmentations=crop,colorjitter       \
     --num_demos 50       \
     --checkpointing_steps 1000       \
     --pretrained_vae_model_name_or_path 'madebyollin/sdxl-vae-fp16-fix' \
     --resume_from_checkpoint 'latest'
```

#### Instruct Pix2Pix with SD Turbo

```bash
cd diffusion
python train_instruct_pix2pix_genima.py
     --pretrained_model_name_or_path='stabilityai/sd-turbo' \
     --output_dir=/tmp/diffusion_agent \
     --resolution=512 \
     --learning_rate=1e-5 \
     --data_path='/tmp/train_data_rgb_rendered/' \
     --validation_images_path '/tmp/train_data_rgb_rendered/'       \
     --validation_prompt 'a robot manipulator taking the lid off a saucepan from wrist perspective'   \
     --train_batch_size=4       \
     --checkpoints_total_limit=2       \
     --num_train_epochs=100       \
     --report_to wandb       \
     --report_name 'pix2pix_turbo_saucepan'       \
     --image_type 'tiled_rgb_rendered'       \
     --conditioning_image_type 'tiled_rgb'       \
     --tasks 'take_lid_off_saucepan'       \
     --validation_steps 500       \
     --mixed_precision='fp16'       \
     --variant='fp16'       \
     --allow_tf32       \
     --enable_xformers_memory_efficient_attention       \
     --tiled       \
     --num_validation_images 4       \
     --augmentations=crop,colorjitter       \
     --num_demos 50       \
     --checkpointing_steps 1000       \
     --resume_from_checkpoint 'latest'
```