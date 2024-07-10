# Model Card: Genima

Following [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993) and [Lessons from Archives (Jo & Gebru)](https://arxiv.org/pdf/1912.10389.pdf) we provide additional information on Genima.

## Model Details


### Overview
- Developed by Shridhar et al. Genima is an end-to-end behavior cloning agent that fine-tunes Stable Diffusion to draw joint-actions on observations. An ACT-based controller is trained from scratch to map target images into a sequence of joint-actions.  
- Architecture: Stable Diffusion uses UNet. ACT uses ResNet-18 vision encoders and Transformer action decoders. 
- Stable Diffusion is fine-tuned to draw joint-actions for tabletop manipulation tasks. ACT is trained with joint targets and random backgrounds.

### Model Date

Aug 2024

### Documents

- [Genima Paper](https://genima-bot.github.io/paper/genima.pdf)
- [Stable Diffusion Turbo Paper](https://arxiv.org/abs/2311.17042)
- [ACT Paper](https://arxiv.org/abs/2304.13705)


## Model Use

- **Primary intended use case**: Genima is intended for robotic manipulation research. We hope the benchmark and pre-trained models will enable researchers to study the capabilities fine-tuned image-generation models for robot control.  
- **Primary intended users**: Robotics researchers. 
- **Out-of-scope use cases**: Deployed use cases in real-world autonomous systems without human supervision during test-time is currently out-of-scope. Use cases that involve manipulating novel objects and observations with people, are not recommended for safety-critical systems. The agent is also intended to be trained and evaluated with English language instructions.

## Data

- Pre-training Data for Stable Diffusion Turbo: see [Model Card](https://huggingface.co/stabilityai/sd-turbo).
- Manipulation Data for Genima: The agent was trained with expert demonstrations. In simulation, we use oracle agents and in real-world we use human demonstrations. Since the agent is used in few-shot settings with very limited data, the agent might exploit intended and un-intented biases in the training demonstrations. 

## Limitations

- Camera extrinsics are needed during training-time.
- Assumes the robot joints are always visible from some camera viewpoint.
- The diffusion agent is slower than the controller.
- Sometimes the controller fails to follow targets provided by the diffusion agent.
- Tasks with extreme object rotation randomization are difficult.
- Genima does not discover new behaviors.

See the Limitations and Potential Solutions section in the [paper](https://genima-bot.github.io/paper/genima.pdf) for an extended discussion.