from agents.dqn_agent import DQNAgent
from utils.train_logger import save_summary

# Train the vanilla DQN agent and Double DQN agent for 1000 episodes
# Remember to change out_file if you don't want to overwrite previous saved models

# dqn_log = DQNAgent().train(1000, train_mode='dqn', print_log=True, out_file='results/dqn')
# ddqn_log = DQNAgent().train(1000, train_mode='ddqn', print_log=True, out_file='results/ddqn')


# Summarize the log and save it to text file
# Remember to change out_file if you don't want to overwrite previous saved table

# save_summary(dqn_log, ddqn_log, 'ddqn', out_file='results/summary_table.txt')


# Load models and test them
DQNAgent().load_and_test('results/dqn_agent.pt', 5, render_mode='human', print_log=True)
DQNAgent().load_and_test('results/ddqn_agent.pt', 5, render_mode='human', print_log=True)