import pandas as pd

from agents.dqn_agent import DQNAgent
from utils.train_logger import save_summary

# Train the vanilla DQN agent and Double DQN agent for 1000 episodes
# Remember to change out_file if you don't want to overwrite previous saved models
# DQNAgent().train(1000, train_mode='dqn', print_log=True, out_file='results/dqn')
# DQNAgent().train(1000, train_mode='ddqn', print_log=True, out_file='results/ddqn')


# Summarize the log and save it to text file
# Remember to change out_file if you don't want to overwrite previous saved table
# dqn_log = pd.read_csv('results/dqn_log.csv')
# ddqn_log = pd.read_csv('results/ddqn_log.csv')
# save_summary(dqn_log, ddqn_log, 'ddqn', out_file='results/summary_table.txt')


# Load models and test them

# With visulaization
# print("Testing DQN...")
# DQNAgent().load_and_test('results/dqn_agent.pt', 5, render_mode='human', print_log=True)
# print("Testing Double DQN...")
# DQNAgent().load_and_test('results/ddqn_agent.pt', 5, render_mode='human', print_log=True)

# With text log only
print("Testing DQN...")
DQNAgent().load_and_test('results/dqn_agent.pt', 100, render_mode=None, print_log=True)
print("Testing Double DQN...")
DQNAgent().load_and_test('results/ddqn_agent.pt', 100, render_mode=None, print_log=True)