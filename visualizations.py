import pandas as pd
import matplotlib.pyplot as plt

dqn_df = pd.read_csv('results/dqn_log.csv')
ext_df = pd.read_csv('results/ddqn_log.csv')

episodes = range(len(dqn_df))

# Episodic Reward vs. Episode Number (for both agents, on the same plot)

plt.figure(figsize=(10, 5))
plt.scatter(episodes, dqn_df['reward'], label='DQN')
plt.scatter(episodes, ext_df['reward'], label='DDQN')
plt.xlabel('Episode')
plt.ylabel('Episodic Reward')
plt.title('Episodic Reward vs Episode')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Episodic Return vs. Episode Number (for both agents, on the same plot)

plt.figure(figsize=(10, 5))
plt.scatter(episodes, dqn_df['return'], label='DQN')
plt.scatter(episodes, ext_df['return'], label='DDQN')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Episodic Return vs Episode')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Success Rate over Time (binary per episode, for both agents, on the same plot)

plt.figure(figsize=(10, 5))
plt.scatter(episodes, dqn_df['success'], label='DQN')
plt.scatter(episodes, ext_df['success'], label='DDQN')
plt.xlabel('Episode')
plt.ylabel('Landing Success')
plt.title('Success per Episode (1 = success, 0 = fail)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

summary = pd.DataFrame({
    'Metric': ['Average Episodic Reward', 'Average Return', 'Success Rate (%)'],
    'DQN (Vanilla)': [
        dqn_df['reward'][-100:].mean(),
        dqn_df['return'][-100:].mean(),
        dqn_df['success'][-100:].mean() * 100
    ],
    'DDQN': [
        ext_df['reward'][-100:].mean(),
        ext_df['return'][-100:].mean(),
        ext_df['success'][-100:].mean() * 100
    ]
})

print(summary)