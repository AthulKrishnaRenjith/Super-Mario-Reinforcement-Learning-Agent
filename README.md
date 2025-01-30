# Super Mario Reinforcement Learning Agent

This project trains a reinforcement learning (RL) agent to play *Super Mario Bros* using Stable Baselines3 and OpenAI Gym. The agent is trained using the PPO (Proximal Policy Optimization) algorithm.

## Features
- Uses `gym_super_mario_bros` to interact with the game environment.
- Implements a custom wrapper (`CustomJoypadSpace`) for improved control.
- Converts game frames to grayscale and stacks them to enhance learning.
- Uses Stable Baselines3's PPO algorithm for training.
- Includes a logging and checkpointing callback.

## Installation
Ensure you have Python installed, then install the required dependencies:

```bash
pip install gym_super_mario_bros nes-py stable-baselines3 matplotlib
```

## Usage
### Training the Agent
Run the script to start training the agent:

```bash
python supermario.ipynb
```

The model will be saved periodically in the `train/` directory.

### Running the Trained Model
After training, you can load and test the model using:

```python
model = PPO.load("thisisatestmodel")

state = env.reset()
while True:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()
    if done:
        state = env.reset()
```

## Hyperparameters
The PPO model is configured with:
- `learning_rate=0.000001`
- `n_steps=512`
- `device="cuda"` (change to "cpu" if CUDA is unavailable)

## Checkpoints and Logging
- Training checkpoints are stored in `./train/`
- TensorBoard logs are stored in `./logs/`

## Notes
- Ensure you have a powerful GPU for faster training.
- The environment renders the game in real-time, which may slow down training.

## Acknowledgement
This project is inspired by Nicholas Renotte's tutorial. You can find his YouTube video here: [Super Mario RL Training](https://youtu.be/2eeYqJ0uBKE?si=wRv-5Hkxe19cS5Tl).

## License
This project is for educational purposes and follows the OpenAI Gym and NES-Py licensing guidelines.


 
