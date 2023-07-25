import argparse
import concurrent.futures
import inspect
import os
import shutil
import sys
from citylearn.agents.marlisa import MARLISA
from citylearn.callback import SaveDataCallback, SaveModelCallback
from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import CustomEVReward, MARL, RewardFunction
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.sac import SAC

def run(func, work_order: list, max_workers: int = None):
    max_workers = 6 if max_workers is None else max_workers

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = [executor.submit(func, **w) for w in work_order]
            
        for i, future in enumerate(concurrent.futures.as_completed(results)):
            try:
                print(f'Completed {i + 1}/{len(results)}:', future.result())
            except Exception as e:
                print(e)
                # raise e

def train_rl_agent(agent_name: str, schema: str, episodes: int, reward_function: RewardFunction, model_directory, results_directory: str, simulation_id: str = None, **kwargs):
    if isinstance(reward_function, str):
        reward_function = {
            'MARL': MARL,
            'CustomEVReward': CustomEVReward
        }[reward_function]
    
    else:
        pass

    default_simulation_id = f'{agent_name}-{schema}-{reward_function.__name__}'.lower()
    simulation_id = default_simulation_id if simulation_id is None else simulation_id

    if agent_name == 'sac':
        func = train_sac_agent
    elif agent_name == 'marlisa':
        func = train_marlisa_agent
    else:
        raise Exception('Unknown agent name:', agent_name)
    
    result = func(
        simulation_id=simulation_id,
        schema=schema,
        episodes=episodes,
        reward_function=reward_function,
        model_directory=model_directory,
        results_directory=results_directory,
        **kwargs
    )

    return result

    
def train_sac_agent(simulation_id: str, schema: dict, episodes: int, reward_function: RewardFunction, model_directory, results_directory: str, **kwargs):
    env = env_creator(schema, reward_function, sb3_model=True)
    model = SAC('MlpPolicy', env, learning_rate=0.005, learning_starts=env.unwrapped.time_steps, tau=0.05, gamma=0.9, **kwargs)
    model_directory = os.path.join(model_directory, simulation_id)
    delete_checkpoint(model_directory)
    checkpoint_callback = CheckpointCallback(
        save_freq=env.time_steps - 1,
        save_path=model_directory,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    save_data_callback = SaveDataCallback(env, simulation_id, results_directory)
    callbacks = [save_data_callback, checkpoint_callback]
    model.learn(total_timesteps=env.time_steps*episodes, callback=callbacks)
    model.save(os.path.join(model_directory, f'{simulation_id}-final.zip'))

    return simulation_id

def train_marlisa_agent(simulation_id: str, schema: dict, episodes: int, reward_function: RewardFunction, model_directory, results_directory: str, **kwargs):
    env = env_creator(schema, reward_function)
    model = MARLISA(env, **kwargs)
    model_directory = os.path.join(model_directory, simulation_id)
    delete_checkpoint(model_directory)
    save_data_callback = SaveDataCallback(env, simulation_id, results_directory)
    save_model_callback = SaveModelCallback(model, simulation_id, model_directory)
    model.learn(
        episodes=episodes, 
        deterministic_finish=True,  
        keep_env_history=False, 
        env_history_directory='env_history',
        save_data_callback=save_data_callback,
        save_model_callback=save_model_callback,
        logging_level=0,
    )
    model_directory = os.path.join(model_directory, simulation_id)
    
    model.save(os.path.join(model_directory, f'{simulation_id}-final.zip'))

    return simulation_id

def env_creator(schema, reward_function: RewardFunction, sb3_model: bool = None) -> CityLearnEnv:
    sb3_model = False if sb3_model is None else sb3_model
    env = CityLearnEnv(schema)
    env.reward_function = reward_function(env)

    if sb3_model:
        env.central_agent = True
        env = NormalizedObservationWrapper(env)
        env = StableBaselines3Wrapper(env)
    else:
        pass

    return env  

def delete_checkpoint(directory: str):
    if os.path.isdir(directory):
        shutil.rmtree(directory)
   
    else:
        pass

def main():
    parser = argparse.ArgumentParser(prog='simulate', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(title='subcommands', required=True, dest='subcommands')
    
    # train
    subparser_train = subparsers.add_parser('train')
    subparser_train.add_argument('agent_name', type=str)
    subparser_train.add_argument('schema', type=str)
    subparser_train.add_argument('episodes', type=int)
    subparser_train.add_argument('reward_function', type=str)
    subparser_train.add_argument('model_directory', type=str)
    subparser_train.add_argument('results_directory', type=str)
    subparser_train.add_argument('-s', '--simulation_id', dest='simulation_id', type=str)
    subparser_train.set_defaults(func=train_rl_agent)

    args = parser.parse_args()
    arg_spec = inspect.getfullargspec(args.func)
    kwargs = {key:value for (key, value) in args._get_kwargs() 
        if (key in arg_spec.args or (arg_spec.varkw is not None and key not in ['func','subcommands']))
    }
    args.func(**kwargs)

if __name__ == '__main__':
    sys.exit(main())