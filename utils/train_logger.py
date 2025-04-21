def train_log(episode, reward, dc_return, success):
    '''
    Print the total reward and discounted return in a specific episode and mark the success episode.
    
    Parameters
    ----------
    episode: int 
        The episode count.
    reward: float
        Total reward.
    dc_return: float
        Discounted return.
    success: int
        1 if the episode found a solution, else 0.
    '''
    success = 'Success' if success == 1 else ''
    print('Episode {} Reward: {} Return: {} {}'.format(str(episode).ljust(7), 
        str(reward).ljust(12), str(dc_return).ljust(12), success))


def test_log(episode, reward, dc_return, steps, success):
    '''
    Print the total reward and discounted return in a specific episode and mark the success episode.
    
    Parameters
    ----------
    episode: int 
        The episode count.
    reward: float
        Total reward.
    dc_return: float
        Discounted return.
    steps: int
        Step taken in this episode.
    success: int
        1 if the episode found a solution, else 0.
    '''
    success = 'Success' if success == 1 else ''
    print('Episode {} Reward: {} Return: {} Steps: {} {}'.format(str(episode).ljust(7), 
        str(reward).ljust(12), str(dc_return).ljust(12), str(steps).ljust(10), success))


def summarize(log):
    '''
    Summarize the result of last 100 episode in the recorded log. The summary includes:
    - Average reward
    - Average return
    - Success rate
    
    Parameters
    ----------
    log: DataFrame
        A dataframe that contains the total reward, discounted return, and whether it successes in each episode.
    
    Returns
    -------
    float, float, float
        Average reward, average return, and success rate in the given order.
    '''
    log = log.tail(100)
    reward = round(log['reward'].mean(), 2)
    return_ = round(log['return'].mean(), 2)
    success = round(log['success'].mean() * 100, 2)
    return reward, return_, success


def save_summary(vanilla_log, extension_log, extension_name, out_file):
    '''
    Create a summary table based on the average reward, average return, and success rate of the vanilla DQN
    and the extension.
    Print the table on save it in a text file.
    
    Parameters
    ----------
    vanilla_log: Dataframe
        A dataframe that contains the total reward, discounted return, and whether it successes in each episode for vanilla DQN.
    extension_log: Dataframe
        A dataframe that contains the total reward, discounted return, and whether it successes in each episode for extension.
    extension_name: str
        The name of the extension.
    out_file: str
        The path of the text file where the output will be stored.
    '''
    
    # Calculate average for vanilla DQN
    vanilla_reward, vanilla_return, vanilla_success = summarize(vanilla_log)
    
    # Calculate average for extension
    extension_reward, extension_return, extension_success = summarize(extension_log)
    
    # Format the summary table
    table = '''Summary over last 100 episodes:
                            {} {}
Average Episodic Reward     {} {}
Average Return              {} {}
Success Rate (%)            {} {}
'''.format('DQN (Vanilla)'.ljust(20), f'DQN + {extension_name.upper()}'.ljust(20),
           str(vanilla_reward).ljust(20), str(extension_reward).ljust(20),
           str(vanilla_return).ljust(20), str(extension_return).ljust(20),
           str(vanilla_success).ljust(20), str(extension_success).ljust(20))

    # Print and save the table
    print(table)
    with open(out_file, "w") as file:
        file.write(table)