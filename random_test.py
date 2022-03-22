import gym
gym.logger.set_level(gym.logger.DEBUG)
import numpy as np
from gym.wrappers.monitor import Monitor
import fetch_block_construction
from gym.wrappers.monitoring import video_recorder

env = gym.make('FetchBlockConstruction_2Blocks_SparseReward_DictstateObs_42Rendersize_FalseStackonly_SingletowerCase-v1')
# env = Monitor(env, directory="videos", force=True, video_callable=lambda x: x)
# vid = video_recorder.VideoRecorder(env,path="./videos/vid.mp4")
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

# TODO: Holding
# TODO: Clear
# TODO: Other remaining predicates


step=0
while True:
    for i in range(50):
        env.render()
        # vid.capture_frame()
        action = np.asarray([0, 0, 0, 0])
        step_results = env.step(action)
        print(on_table(env, 'object0'))
        print(on_table(env, 'object1'))
        obs, reward, done, info = step_results
        print("Reward: {} Info: {}".format(reward, info))
        if done:
            step = 0
            import ipdb; ipdb.set_trace()
            env.reset()
            # vid.close()
            break
        step+=1
        print(step)

# Getting some x-position: env.sim.data.get_site_xpos('object0')
# Get contacts:
# for i in range(env.sim.data.ncon):
#     contact = env.sim.data.contact[i]
#     print('contact', i)
#     print('geom1', contact.geom1, env.sim.model.geom_id2name(contact.geom1))
#     print('geom2', contact.geom2, env.sim.model.geom_id2name(contact.geom2))
