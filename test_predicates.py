import gym
gym.logger.set_level(gym.logger.DEBUG)
import numpy as np
from gym.wrappers.monitor import Monitor
import fetch_block_construction
from gym.wrappers.monitoring import video_recorder
import copy
from utils import Predicates

NUM_BLOCKS = 2

env = gym.make('FetchBlockConstruction_2Blocks_SparseReward_DictstateObs_42Rendersize_FalseStackonly_SingletowerCase-v1')
# env = Monitor(env, directory="videos", force=True, video_callable=lambda x: x)
# vid = video_recorder.VideoRecorder(env,path="./videos/vid.mp4")
env.env._max_episode_steps = 50

obs = env.reset()
env.env.stack_only = True

def print_predicates(env):
    predicates = Predicates().get_predicates()
    objects = ['floor0', 'table0'] + [f"object{i}" for i in range(NUM_BLOCKS)]
    for predicate in predicates["0-arity"]:
        print(predicate.__name__, predicate(env))

    for predicate in predicates["1-arity"]:
        for obj in objects:
            if obj not in ['floor0', 'table0']:
                print(predicate.__name__, [obj], predicate(env, obj))
    
    for predicate in predicates["2-arity"]:
        for obj1 in objects:
            for obj2 in objects:
                if obj1 != obj2:
                    print(predicate.__name__, [obj1, obj2], predicate(env, obj1, obj2))  


def set_object_pos(env, obj_name, pos):
    object_pos = pos
    object_qpos = env.sim.data.get_joint_qpos(obj_name + ':joint')
    assert object_qpos.shape == (7,)
    object_qpos[:3] = object_pos
    env.sim.data.set_joint_qpos(obj_name + ':joint', object_qpos)

def set_gripper_pos(env, pos):
    gripper_target = pos
    gripper_rotation = np.array([1., 0., 1., 0.])
    env.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
    env.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
    for _ in range(10):
        env.sim.step()

# Starts obj1 on obj0 and runs simple policy that grabs obj1 to check predicates
step=0
env._max_episode_steps = 1000
pos = np.array([1.5, 0.8, 0.4])
set_object_pos(env, 'object0', pos)
pos = np.array([1.5, 0.8, 0.75])
set_object_pos(env, 'object1', pos)
gripper_pos = np.array([1.5, 0.8, 0.8])
set_gripper_pos(env, gripper_pos)
grasp_count = 0
policy_part = 0
x_drop_delta = -0.1 #0 if ontop -0.1 otherwise

for run in range(2):
    for i in range(50000):
        env.render()
        # vid.capture_frame()
        gripper_pos = env.sim.data.get_site_xpos('robot0:grip')
        obj1_pos = env.sim.data.get_joint_qpos('object0:joint')
        obj2_pos = env.sim.data.get_joint_qpos('object1:joint')
        if policy_part == 0:
            action_delta = obj2_pos[:3] - gripper_pos
        else:
            action_delta = obj1_pos[:3] - gripper_pos
        action = np.concatenate([action_delta, [1]])
        if action[:3].dot(action[:3]) < 0.001:
            action[3] = -1
            grasp_count += 1
        if grasp_count > 20:
            if grasp_count < 100:
                action = np.array([x_drop_delta, 0, 0.1, -1])
                grasp_count += 1
            elif grasp_count < 150:
                action = np.array([0, 0, 0, 0.5])
                grasp_count += 1
            else:
                policy_part = 1
                grasp_count = 0

        #print(policy_part, grasp_count, action)
        
        step_results = env.step(action)
        obs, reward, done, info = step_results
        #print(obs)

        #import ipdb; ipdb.set_trace()

        print_predicates(env)
        print()

        if done:
            step = 0
            env.reset()
            pos = np.array([1.5, 0.8, 0.4])
            set_object_pos(env, 'object0', pos)
            pos = np.array([1.5, 0.8, 0.75])
            set_object_pos(env, 'object1', pos)
            gripper_pos = np.array([1.5, 0.8, 0.8])
            set_gripper_pos(env, gripper_pos)
            # vid.close()
            break
        step+=1

# Getting some x-position: env.sim.data.get_site_xpos('object0')
# Get contacts:
# for i in range(env.sim.data.ncon):
#     contact = env.sim.data.contact[i]
#     print('contact', i)
#     print('geom1', contact.geom1, env.sim.model.geom_id2name(contact.geom1))
#     print('geom2', contact.geom2, env.sim.model.geom_id2name(contact.geom2))
