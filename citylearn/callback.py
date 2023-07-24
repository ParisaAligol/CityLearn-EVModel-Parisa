import os
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

class SaveDataCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, env, simulation_id, simulation_output_path, extras=None, verbose=0):
        super(SaveDataCallback, self).__init__(verbose)
        self.env = env
        self.simulation_id = simulation_id
        self.simulation_output_path = simulation_output_path
        self.episode = 0
        self.extras = {} if extras is None else extras
        self.mode = 'train'

    def _on_step(self) -> bool:
        # save timer data
        if self.env.time_step == self.env.time_steps - 2:
            save_data(
                self.env, 
                self.simulation_id, 
                self.simulation_output_path, 
                self.episode, 
                self.mode,
                self.extras
            )
            self.episode += 1

        else:
            pass

        return True
    
def save_data(env, simulation_id, simulation_output_path, episode, mode, extras):    
    # save reward data
    reward_data = pd.DataFrame(env.rewards, columns=['reward'])
    reward_data['time_step'] = reward_data.index
    reward_data['building_name'] = None
    reward_data['mode'] = mode
    reward_data['episode'] = episode
    reward_data['simulation_id'] = simulation_id

    for k, v in extras.items():
        reward_data[k] = v

    reward_filepath = os.path.join(simulation_output_path, f'reward-{simulation_id}.csv')

    if episode > 0 and os.path.isfile(reward_filepath):
        existing_data = pd.read_csv(reward_filepath)
        reward_data = pd.concat([existing_data, reward_data], ignore_index=True, sort=False)
        del existing_data
    else:
        pass

    reward_data.to_csv(reward_filepath, index=False)
    del reward_data

    # save KPIs
    ## building level
    kpi_data = env.evaluate()
    kpi_data['mode'] = mode
    kpi_data['episode'] = episode
    kpi_data['simulation_id'] = simulation_id

    for k, v in extras.items():
        kpi_data[k] = v

    kpi_filepath = os.path.join(simulation_output_path, f'kpis-{simulation_id}.csv')

    if episode > 0 and os.path.isfile(kpi_filepath):
        existing_data = pd.read_csv(kpi_filepath)
        kpi_data = pd.concat([existing_data, kpi_data], ignore_index=True, sort=False)
        del existing_data
    else:
        pass
 
    kpi_data = kpi_data.dropna()
    kpi_data.to_csv(kpi_filepath, index=False)
    del kpi_data