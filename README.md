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
```

After your computer is ready, to run the code, execute the following command:

```bash
python main.py
```
You will see printed results for each episode showing:
- Reward
- Return
- Steps taken
- Success or failure

Example output:
```python
Testing model in file: results/dqn_agent.pt
Episode 1       Reward: 284.96       Return: 74.01        Steps: 234        Success
Episode 2       Reward: 220.68       Return: 45.75        Steps: 418        Success
Episode 3       Reward: 274.84       Return: 85.66        Steps: 252        Success
Episode 4       Reward: 101.99       Return: 37.75        Steps: 1000       
Episode 5       Reward: 268.28       Return: 65.23        Steps: 265        Success
Testing model in file: results/ddqn_agent.pt
Episode 1       Reward: 272.01       Return: 57.11        Steps: 311        Success
Episode 2       Reward: 255.02       Return: 54.47        Steps: 270        Success
Episode 3       Reward: 278.54       Return: 76.28        Steps: 237        Success
Episode 4       Reward: 224.62       Return: 45.76        Steps: 249        Success
Episode 5       Reward: 228.56       Return: 49.13        Steps: 331        Success
```

##Hyperparameters in the Source Code

The specific values for the hyperparameters that control training behavior are set up directly in the source code (dqn_agent.py)

Here are the important ones:

```python
LEARNING_RATE = 0.0001
DISCOUNT_FACTOR = 0.99
EPSILON_DECAY = 500
```

These values are defined as constants at the top of DQNAgent class. If you wish to experiment with different values, you can manually modify these constants inside dqn_agent.py

For example:
```python
LR = 0.001         # Increase learning rate
DF = 0.95          # Discount future rewards a bit more
EPS_DECAY = 800    # Decay epsilon more slowly
```
Feel free to adjust these values or add your own. Whatever helps your agent perform better!
