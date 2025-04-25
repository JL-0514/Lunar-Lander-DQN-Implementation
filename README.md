# Lunar-Lander-DQN-Implementation

Implementation of a Deep Q-Network (DQN) to solve the LunarLander-v3 environment from Gymnasium.

## Contributors:
- Zane Swaims
- Jayne Nemynova
- Emily Zapata
- Cathy Che
- Jiameng Li

## Requirements:
In order to run this project, you need Python3.10 and following libraries:
- PyTorch
- Gymnasium
- LunarLander-v3

## Instructions:
To install all required libraries, activate your virtual environment and run:

```bash
pip install "gymnasium[box2d]" torch numpy matplotlib seaborn pygame


After your computer is ready, to run the code, execute the following command:

```bash
python main.py

You will see printed results for each episode showing:
- Reward
- Return
- Steps taken
- Success or failure

Example output:
![example output](pics/example_output.png)


Hyperparameters in the Source Code

The specific values for the hyperparameters that control training behavior are set up directly in the source code (dqn_agent.py)

Here are the important ones:

LEARNING_RATE = 0.0001
DISCOUNT_FACTOR = 0.99
EPSILON_DECAY = 500

These values are defined as constants at the top of DQNAgent class. If you wish to experiment with different values, you can manually modify these constants inside dqn_agent.py

For example:

LR = 0.001         # Increase learning rate
DF = 0.95          # Discount future rewards a bit more
EPS_DECAY = 800    # Decay epsilon more slowly

Feel free to adjust these values or add your own. Whatever helps your agent perform better!
