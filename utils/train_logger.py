def episode_log(episode, reward, dc_return, success):
    success = 'success' if success == 1 else ''
    print('Episode {} reward {} return {} {}'.format(str(episode).ljust(7), 
        str(reward).ljust(12), str(dc_return).ljust(12), success))


def save_avg(vanilla_log, extension_log, extension_name, put_path='results/'):
    
    # Calculate average for vanilla DQN
    vanilla_reward = round(vanilla_log['reward'].mean(), 2)
    vanilla_return = round(vanilla_log['return'].mean(), 2)
    vanilla_success = round(vanilla_log['success'].mean() * 100, 2)
    
    # Calculate average for extension
    extension_reward = round(extension_log['reward'].mean(), 2)
    extension_return = round(extension_log['return'].mean(), 2)
    extension_success = round(extension_log['success'].mean() * 100, 2)
    
    # Save average table
    table = f'''
                            DQN (Vanilla)       DQN + {extension_name.upper()}
Average Episodic Reward     {vanilla_reward}    {extension_reward}
Average Return              {vanilla_return}    {extension_return}
Success Rate (%)            {vanilla_success}   {extension_success}
'''
    print(table)
    with open(f"{put_path}average_table.txt", "w") as file:
        file.write(table)