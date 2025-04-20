from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    '''
    The Deep Q-learning Network module that take the state (observations) of the agent and 
    calculate the best action.
    '''
    
    def __init__(self, n_observations, n_actions):
        '''
        Initialize a DQN that take a certain number of observations and select the best action
        from a certain number of possible actions.
        
        Parameters
        ----------
        n_observations: int
            Number of observations for input.
        n_actions: int
            Number of possible actions.
        '''
        super(DQN, self).__init__()
        # Neural network
        # TODO Improve the module by modify hidden layers
        # For now, it has a hidden layer with 128 nodes
        self.input = nn.Linear(n_observations, 128)
        self.output = nn.Linear(128, n_actions)
    
    
    def forward(self, x):
        '''
        Define the computation performed at every call.
        The module will call this function automatically.
        '''
        x = F.relu(self.input(x))
        return self.output(x)