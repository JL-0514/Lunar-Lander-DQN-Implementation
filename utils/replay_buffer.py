from collections import deque
import random

class ReplayBuffer():
    '''
    The replay buffer store the transition made by the agent in each step.
    Each transition contains list of information [current_state, action, next_state, reward, is_done] 
    in the given order.
    - current_state (observation) is the state before action.
    - action (int) is one of possible actions the agent can take.
    - next_state (observation) is the state after action.
    - reward (float) is the reward earned after taking the action in current_state.
    - is_done (boolean) is whether the agent should stop acting in next state.
    
    The observation contains list of information [x-coordinate, y-coordinate, x-velocity, y-velocity, 
    angle, angular velocity, leg 1 on ground (boolean), leg 2 on ground (boolean)].
    
    When the number of transitions reach its capacity, it will drop the oldest transition.
    
    Attributes
    ----------
    memory: deque
        A double-ended queue used to store transition.
    
    Methods
    -------
    store(capacity)
        Store the given transition in the buffer.
    sample(batch_size)
        Get a list of random sample of transition with given batch size.
    '''
    
    def __init__(self, capacity: int):
        '''
        Initialize a replay buffer with the given capacity.
        
        Parameters
        ----------
        - capacity: int
            Capacity of the buffer
        '''
        self.memory = deque([], maxlen=capacity)
    
    
    def store(self, transition: list):
        '''
        Store the given transition in the buffer.
        
        Parameters
        ----------
        - transition: list
            The transition the contains list of information [current_state, action, next_state, reward, is_done] 
            in the given order.
        '''
        self.memory.append(transition)


    def sample(self, batch_size: int):
        '''
        Get a list of random sample of transition with given batch size.
        
        Parameters
        ----------
        - batch_size: int
            The size of sample needed.
            
        Returns
        -------
        list
            A list of random sample of transition.
        '''
        return random.sample(self.memory, batch_size)


    def __len__(self):
        return len(self.memory)