import gym
gym.logger.set_level(gym.logger.DEBUG)
import numpy as np
import fetch_block_construction  # NOTE: Necessary import for env to be created even though not directly used!
import torch
import os
import torch.nn as nn
from pathlib import Path

from easyrl.agents.ppo_agent import PPOAgent
from easyrl.configs import cfg
from easyrl.configs import set_config
from easyrl.configs.command_line import cfg_from_cmd
from easyrl.engine.ppo_engine import PPOEngine
from easyrl.models.categorical_policy import CategoricalPolicy
from easyrl.models.diag_gaussian_policy import DiagGaussianPolicy
from easyrl.models.mlp import MLP
from easyrl.models.value_net import ValueNet
from easyrl.runner.nstep_runner import EpisodicRunner
from easyrl.utils.common import set_random_seed
from easyrl.utils.gym_util import make_vec_env


NUM_BLOCKS = 2
env = gym.make('FetchBlockConstruction_2Blocks_SparseReward_DictstateObs_42Rendersize_FalseStackonly_SingletowerCase-v1')
env.env._max_episode_steps = 50

obs = env.reset()
env.env.stack_only = True

def on_table(env, obj):
    # Get table z-coordinate
    table_dim = [0.25, 0.35, 0.2]
    table_id = env.sim.model.body_name2id('table0')
    table_pos = env.sim.model.body_pos[table_id]
    table_pos_min = table_pos - table_dim
    table_pos_max = table_pos + table_dim
    table_z = table_pos_max[-1]    
    # Get object z-coordinate
    object_height = 0.025
    object_pos = env.sim.data.get_site_xpos(obj)
    object_base_z = object_pos[-1] - object_height
    
    if (table_pos_min[0] <= object_pos[0] <= table_pos_max[0] and table_pos_min[1] <= object_pos[1] <= table_pos_max[1] and abs(table_z - object_base_z) <= 1e-3):
        return True
    else:
        return False

def in_contact(env, obj1_name, obj2_name):
    """Helper function that simply tests whether 2 objects are in contact with each other"""
    for i in range(env.sim.data.ncon):
        contact = env.sim.data.contact[i]
        geom1_name = env.sim.model.geom_id2name(contact.geom1)
        geom2_name = env.sim.model.geom_id2name(contact.geom2)
        if (geom1_name == obj1_name and geom2_name == obj2_name) or (geom1_name == obj2_name and geom2_name == obj1_name):
            return True
    return False

def ontable_contact(env, obj_name):
    """Test this predicate by simply checking whether the object in question is touching the table."""
    return in_contact(env, obj_name, "table0")

def on(env, obj1_name, obj2_name):
    """Test whether obj1 is on obj2 by checking if they are in contact, and one object's z pose is higher than the other"""
    obj1_id = env.sim.model.body_name2id(obj1_name)
    obj2_id = env.sim.model.body_name2id(obj2_name)
    obj1_pos = env.sim.model.body_pos[obj1_id]
    obj2_pos = env.sim.model.body_pos[obj2_id]
    if obj1_pos[-1] <= obj2_pos[-1]:
        return False
    return in_contact(obj1_name, obj2_name)

def clear(env, obj_name):
    """Test whether obj_name has nothing on it."""
    for i in range(NUM_BLOCKS):
        curr_block_name = f"object{i}"
        if curr_block_name == obj_name:
            continue
        else:
            if on(env, curr_block_name, obj_name):
                return False
    return True

def r_gripper_contact(env, obj_name):
    return in_contact(env, obj_name, 'robot0:r_gripper_finger_link')

def l_gripper_contact(env, obj_name):
    return in_contact(env, obj_name, 'robot0:l_gripper_finger_link')

def holding(env, obj_name):
    return r_gripper_contact(env, obj_name) and l_gripper_contact(env, obj_name)

def hand_empty(env):
    for i in range(NUM_BLOCKS):
        curr_block_name = f"object{i}"
        if holding(env, curr_block_name):
            return False 
    return True


def main():
    set_config('ppo')
    cfg_from_cmd(cfg.alg)
    if cfg.alg.env_name is None:
        # cfg.alg.env_name = 'FetchBlockConstruction_2Blocks_SparseReward_DictstateObs_42Rendersize_FalseStackonly_SingletowerCase-v1'
        cfg.alg.env_name = 'FetchBlockConstruction_2Blocks_DenseReward_DictstateObs_42Rendersize_FalseStackonly_SingletowerCase-v1'
    if cfg.alg.resume or cfg.alg.test:
        if cfg.alg.test:
            skip_params = [
                'test_num',
                'num_envs',
                'sample_action',
            ]
        else:
            skip_params = []
        cfg.alg.restore_cfg(skip_params=skip_params)#,path=Path(os.path.join(os.getcwd(),'data', cfg.alg.env_name)))

    if torch.cuda.is_available():
        cfg.alg.device = 'cuda'
    else:
        cfg.alg.device = 'cpu'
    set_random_seed(cfg.alg.seed)
    if cfg.alg.test:
        cfg.alg.num_envs = 1
    env = make_vec_env(cfg.alg.env_name,
                    cfg.alg.num_envs,
                    seed=cfg.alg.seed)
    env.reset()
    ob_size = env.observation_space['observation'].shape[0]

    actor_body = MLP(input_size=ob_size,
                     hidden_sizes=[64],
                     output_size=64,
                     hidden_act=nn.ReLU,
                     output_act=nn.ReLU)

    critic_body = MLP(input_size=ob_size,
                      hidden_sizes=[64],
                      output_size=64,
                      hidden_act=nn.ReLU,
                      output_act=nn.ReLU)
    if isinstance(env.action_space, gym.spaces.Discrete):
        act_size = env.action_space.n
        actor = CategoricalPolicy(actor_body,
                                  in_features=64,
                                  action_dim=act_size)
    elif isinstance(env.action_space, gym.spaces.Box):
        act_size = env.action_space.shape[0]
        actor = DiagGaussianPolicy(actor_body,
                                   in_features=64,
                                   action_dim=act_size,
                                   tanh_on_dist=cfg.alg.tanh_on_dist,
                                   std_cond_in=cfg.alg.std_cond_in)
    else:
        raise TypeError(f'Unknown action space type: {env.action_space}')

    critic = ValueNet(critic_body, in_features=64)
    
    agent = PPOAgent(actor=actor, critic=critic, env=env)
    runner = EpisodicRunner(agent=agent, env=env)
    engine = PPOEngine(agent=agent,
                       runner=runner)
    if not cfg.alg.test:
        engine.train()
    else:
        stat_info, raw_traj_info = engine.eval(render=cfg.alg.render,
                                               save_eval_traj=cfg.alg.save_test_traj,
                                               eval_num=cfg.alg.test_num,
                                               sleep_time=0.04)
        import pprint
        pprint.pprint(stat_info)
    env.close()


if __name__ == '__main__':
    main()

# NOTE:
# To train, run: `python run_ppo.py`` from the root of this repository
# To eval, run: `python run_ppo.py --render --test` from the root of this repository
# Also, the reward function is coming from `compute_reward()` in construction.py