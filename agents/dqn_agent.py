import gymnasium as gym
import pandas as pd
import numpy as np 
import random
import torch
import math

from utils.replay_buffer import ReplayBuffer
from utils.train_logger import episode_log
from networks.q_network import DQN

class DQNAgent():
    '''
    The DQN agent used to solve LunarLander-v3 environment.
    
    Methods
    -------
    train(episodes, render_mode=None, train_mode="dqn")
        Train the agent by given number of episodes, render mode, and training mode.
    '''
    
    # Constants need for training
    # TODO Changing parameter to improve performance
    LR = 1e-4           # Learning rate (alpha)
    DF = 0.99           # Discuont factor (gamma)
    BATCH_SIZE = 128    # Batch size for sampling transition
    CAPACITY = 40000    # Capacity of replay buffer
    UR = 20             # Update rate for target network
    EPS_START = 1       # The initial value of epsilon
    EPS_END = 0.05      # The final value of epsilon
    EPS_DECAY = 500     # Controls the rate of exponential decay of epsilon
        
    # Loss function
    # LOSS_F = torch.nn.MSELoss()
    LOSS_F = torch.nn.SmoothL1Loss()
    
    
    def train(self, episodes, render_mode=None, train_mode="dqn", out_path='results/'):
        '''
        Train the agent by given number of episodes, render mode, and training mode.
        The trained agent and the training log will be saved in a specific folder.
        
        Parameters
        ----------
        episodes: int
            Number of episodes the agent will go through before ending the train.
        render_mode: str
            In which way will the environment render. Available choices are None, 'human', and 'rgb-array'.
        train_mode: str
            Whether the agent is trained with vanilla DQN or extension. Available choices are 'dqn' and 'ddqn' where ddqn stands for double dqn.
        out_path: str
            The path to the folder where the trained agent and training log will be saved.
        
        Returns
        ------
        DataFrame
            A dataframe that contains total reward, discount return, whether agent found a solution in each episode.
        '''
        
        # Create the LunarLander-v3 environment
        env = gym.make("LunarLander-v3", render_mode=render_mode)
        
        # Find the size of observation and action space.
        n_actions = env.action_space.n
        state, info = env.reset()
        n_observations = len(state)
        
        # Initialize epsilon to 1; the action is selected 100% random at the beginning
        epsilon = self.EPS_START
        
        # initialize a matrix to keep track of episodic reward, discounted return, 
        # and whether the lander land successfully in each episode (1 if sucess, else 0).
        log_matrix = pd.DataFrame(columns=['reward', 'return', 'success'])
        
        # Apply Deep Q-learning Algorithm:
        
        # Initialize replay buffer
        buffer = ReplayBuffer(self.CAPACITY)
        
        # Initialize policy network and a target network
        policy_net = DQN(n_observations, n_actions)
        target_net = DQN(n_observations, n_actions)
        
        # Initialize a gradient descent optimizer
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=self.LR)
        
        print(f"Train mode: {train_mode}")
        
        # Start the loop in range 1 to episodes:
        for episode in range(episodes):
            
            # Reset the state of the environment to start a new episode
            state, info = env.reset()
            total_reward = 0        # Reward earned in current episode
            discounted_return = 0   # Return earned in current episode
            
            # Limit the number of steps it can take within an episode
            # TODO Change the limit may improve performance
            for step in range(1000):
                
                # select an action; get a random number
                if random.random() < epsilon:   # Get a random number, if it's less than epsilon, select random action
                    action = env.action_space.sample()
                else:   # Else select the best action by applying current state to policy network
                    with torch.no_grad():
                        action = policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax().item()

                
                # Make the agent take the selected action and get the reward and next state
                next_state, reward, terminated, truncated, info = env.step(action)
                
                # Store the transition to the replay buffer
                buffer.store([state, action, next_state, reward, terminated or truncated])
                state = next_state
                
                # Record reward and discounted return
                total_reward += reward
                discounted_return += self.DF ** step * reward
                
                # Get sample transitions from the replay_buffer
                if len(buffer) > self.BATCH_SIZE:
                    batch = buffer.sample(self.BATCH_SIZE)
                    
                    # Recall the elements in transition are [current_state, action, next_state, reward, is_done]
                    states = torch.from_numpy(np.array([x[0] for x in batch])).float()
                    actions = torch.tensor([x[1] for x in batch], dtype=torch.long)
                    next_states = torch.from_numpy(np.array([x[2] for x in batch])).float()
                    rewards = torch.tensor([x[3] for x in batch], dtype=torch.float32)
                    dones = torch.tensor([x[4] for x in batch], dtype=torch.bool)
                    
                    # Get new Q values of each transition from the target network Q'
                    # new_Q_value = reward + discount_factor * Q'(next_state) or reward if is_done
                    with torch.no_grad():
                        target_q = rewards + self.DF * target_net(next_states).max(1)[0] * (~dones)
                    
                    # Get old Q values of each transition from the policy network Q
                    # old_Q_value = Q(current_state)
                    policy_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    
                    # Compute the loss
                    loss = self.LOSS_F(policy_q, target_q)
                    # Perform gradient descent step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # Check if we need to update the target network
                if step % self.UR == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                    
                # Visualize the training
                if render_mode is not None:
                    still_open = env.render()
                    if still_open is False:
                        break
                
                # If the agent is terminated or truncated, continue to next episode
                if terminated or truncated:
                    break
                
            # Decay epsilon at the end of each epsilon
            epsilon = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * episode / self.EPS_DECAY)
        
            # Record training log
            success = 1 if total_reward >= 200 else 0
            total_reward = round(total_reward, 2)
            discounted_return = round(discounted_return, 2)
            log_matrix.loc[len(log_matrix)] = [total_reward, discounted_return, success]
            episode_log(episode, total_reward, discounted_return, success)
        
        # Close the environment
        env.close()
        
        # Save the trained agent
        torch.save(policy_net.state_dict(), f"{out_path}agent_{train_mode}.pt")
        
        # Save the training log
        log_matrix.to_csv(f'{out_path}log_{train_mode}.csv', index=False)
        
        return log_matrix