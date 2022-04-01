import gym
import copy
import numpy as np
import fetch_block_construction
gym.logger.set_level(gym.logger.DEBUG)
from gym.wrappers.monitor import Monitor
from gym.wrappers.monitoring import video_recorder


# Current implementation restricts goals for 2 object stacking only.
NUM_BLOCKS = 2
env = gym.make('FetchBlockConstruction_2Blocks_SparseReward_DictstateObs_42Rendersize_FalseStackonly_SingletowerCase-v1')
env.env._max_episode_steps = 50

obs = env.reset()
env.env.stack_only = True

class Predicates:

    def on_table(self, env, obj):
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

    def in_contact(self, env, obj1_name, obj2_name):
        """Helper function that simply tests whether 2 objects are in contact with each other"""
        for i in range(env.sim.data.ncon):
            contact = env.sim.data.contact[i]
            geom1_name = env.sim.model.geom_id2name(contact.geom1)
            geom2_name = env.sim.model.geom_id2name(contact.geom2)
            if (geom1_name == obj1_name and geom2_name == obj2_name) or (geom1_name == obj2_name and geom2_name == obj1_name):
                return True
        return False

    def ontable_contact(self, env, obj_name):
        """Test this predicate by simply checking whether the object in question is touching the table."""
        return self.in_contact(env, obj_name, "table0")

    def on(self, env, obj1_name, obj2_name):
        """Test whether obj1 is on obj2 by checking if they are in contact, and one object's z pose is higher than the other"""
        if "object" in obj1_name:
            obj1_pos = env.sim.data.get_joint_qpos(obj1_name + ':joint')
            obj1_z = obj1_pos[2]
        else:
            obj1_id = env.sim.model.body_name2id(obj1_name)
            obj1_pos = env.sim.model.body_pos[obj1_id]
            obj1_z = obj1_pos[-1]

        if "object" in obj2_name:
            obj2_pos = env.sim.data.get_joint_qpos(obj2_name + ':joint')
            obj2_z = obj2_pos[2]
        else:
            obj2_id = env.sim.model.body_name2id(obj2_name)
            obj2_pos = env.sim.model.body_pos[obj2_id]
            obj2_z = obj2_pos[-1]
            
        if obj1_z <= obj2_z:
            return False
        return self.in_contact(env, obj1_name, obj2_name)

    def clear(self, env, obj_name):
        """Test whether obj_name has nothing on it."""
        for i in range(NUM_BLOCKS):
            curr_block_name = f"object{i}"
            if curr_block_name == obj_name:
                continue
            else:
                if self.on(env, curr_block_name, obj_name):
                    return False
        return True

    def r_gripper_contact(self, env, obj_name):
        return self.in_contact(env, obj_name, 'robot0:r_gripper_finger_link')

    def l_gripper_contact(self, env, obj_name):
        return self.in_contact(env, obj_name, 'robot0:l_gripper_finger_link')

    def holding(self, env, obj_name):
        return self.r_gripper_contact(env, obj_name) and self.l_gripper_contact(env, obj_name)

    def hand_empty(self, env):
        for i in range(NUM_BLOCKS):
            curr_block_name = f"object{i}"
            if self.holding(env, curr_block_name):
                return False 
        return True

    def get_predicates(self):
        return {"0-arity": [self.hand_empty], "1-arity": [self.on_table, self.clear, self.holding], "2-arity": [self.on]}

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

print_predicates(env)


predicates = Predicates().get_predicates()
objects = [f"object{i}" for i in range(NUM_BLOCKS)]
problem = "(define (problem task)\n\t(:domain blocksworld)\n\t(:objects "

for i in range(NUM_BLOCKS):
    problem += objects[i]
    if (i != NUM_BLOCKS - 1):
        problem += " "

problem += ")\n\t(:init\n\t\t"

for predicate in predicates["0-arity"]:
    if (predicate(env)):
        problem += "(" + predicate.__name__ + ")\n\t\t"

for predicate in predicates["1-arity"]:
    for obj in objects:
        if (predicate(env, obj)):
            problem += "(" + predicate.__name__ + " " + obj + ")\n\t\t"

for predicate in predicates["2-arity"]:
    for obj1 in objects:
        for obj2 in objects:
            if (obj1 != obj2 and predicate(env, obj1, obj2)):
                problem += "(" + predicate.__name__ + " " + obj1 + " " + obj2 + ")\n\t\t"

problem += "\n\t)\n\t(:goal (and \n\t\t(on object1 object0))\n\t)\n)"

with open("problem.pddl", "w") as f:
    f.write(problem)