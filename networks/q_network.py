from torch import nn

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
        # TODO Improve the module by modifying hidden layers
        self.net = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    
    
    def forward(self, x):
        '''
        Define the computation performed at every call.
        The module will call this function automatically.
        '''
        return self.net(x)