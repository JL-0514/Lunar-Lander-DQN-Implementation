import pandas as pd
import matplotlib.pyplot as plt

# Mounting the data for dqn and ddqn
dqn_df = pd.read_csv('results/dqn_log.csv')
ext_df = pd.read_csv('results/ddqn_log.csv')

# Creating a value that will be used to actually plot
episodes = range(len(dqn_df))
WIDTH = 20
HEIGHT = 5

# Episodic Reward vs. Episode Number (for both agents, on the same plot)

# Creating and graphing the figure using matplotlib
# Setting the default size, so it is readable out of the bat
plt.figure(figsize=(WIDTH, HEIGHT))

# Actually plotting the scatter points for the dqn and ddqn data sets
plt.scatter(episodes, dqn_df['reward'], label='DQN')
plt.scatter(episodes, ext_df['reward'], label='DDQN')

# Labeling the axis' and the graph for readability
plt.xlabel('Episode')
plt.ylabel('Episodic Reward')
plt.title('Episodic Reward vs Episode')

# Printing out a legend and a grid to identify what the user is looking at
plt.legend()
plt.grid(True)

# Prettying up the layout and saving it
plt.tight_layout()
# plt.show()
plt.savefig('results/episodic_reward_vs_episode.png', dpi=300)

# Episodic Return vs. Episode Number (for both agents, on the same plot)

# The process for the next two graphs is almost the exact same
plt.figure(figsize=(WIDTH, HEIGHT))
plt.scatter(episodes, dqn_df['return'], label='DQN')
plt.scatter(episodes, ext_df['return'], label='DDQN')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Episodic Return vs Episode')
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig('results/episodic_return_vs_episode.png', dpi=300)

# Success Rate over Time (binary per episode, for both agents, on the same plot)

plt.figure(figsize=(WIDTH, HEIGHT))
plt.scatter(episodes, dqn_df['success'], label='DQN')
plt.scatter(episodes, ext_df['success'], label='DDQN')
plt.xlabel('Episode')
plt.ylabel('Landing Success')
plt.title('Success per Episode (1 = success, 0 = fail)')
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig('results/success_per_episode.png', dpi=300)

# Doing some basic math to display the statistics asked of us
summary = pd.DataFrame({
    'Metric': ['Average Episodic Reward', 'Average Return', 'Success Rate (%)'],
    'DQN': [
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

# print(summary)

with open("results/summary.md", "w") as f:
    f.write(summary.to_markdown(index=False))