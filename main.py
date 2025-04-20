from agents.dqn_agent import DQNAgent
from utils.train_logger import save_avg

# Declare a DQN agent
agent = DQNAgent()

# Train the agent with vanilla DQN
dqn_log = agent.train(1500)