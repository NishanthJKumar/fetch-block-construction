
import gym
gym.logger.set_level(gym.logger.DEBUG)
import numpy as np
from gym.wrappers.monitor import Monitor
import fetch_block_construction
from gym.wrappers.monitoring import video_recorder
import copy
import pddlpy

domprob = pddlpy.DomainProblem('blocksworld.pddl', 'problem.pddl')
NUM_BLOCKS = 2

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
        if "object" in obj1_name and "object" in obj2_name:
            obj1_pos = env.sim.data.get_joint_qpos(obj1_name + ':joint')
            obj2_pos = env.sim.data.get_joint_qpos(obj2_name + ':joint')
            if obj1_pos[2] <= obj2_pos[2]:
                return False
            return self.in_contact(env, obj1_name, obj2_name)
        else:
            obj1_id = env.sim.model.body_name2id(obj1_name)
            obj2_id = env.sim.model.body_name2id(obj2_name)
            obj1_pos = env.sim.model.body_pos[obj1_id]
            obj2_pos = env.sim.model.body_pos[obj2_id]
            if obj1_pos[-1] <= obj2_pos[-1]:
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


def get_state_grounded_atoms(env):
    state_grounded_atoms = []

    predicates = Predicates().get_predicates()
    objects = ['floor0', 'table0'] + [f"object{i}" for i in range(NUM_BLOCKS)]
    for predicate in predicates["0-arity"]:
        state_grounded_atoms.append([(predicate.__name__,), predicate(env)])

    for predicate in predicates["1-arity"]:
        for obj in objects:
            if obj not in ['floor0', 'table0']:
                state_grounded_atoms.append([(predicate.__name__, obj), predicate(env, obj)])
    
    for predicate in predicates["2-arity"]:
        for obj1 in objects:
            for obj2 in objects:
                if obj1 != obj2:
                    state_grounded_atoms.append([(predicate.__name__, obj1, obj2), predicate(env, obj1, obj2)]) 

    return [atom[0] for atom in state_grounded_atoms if atom[1]]

def apply_grounded_operator(state_grounded_atoms, op_name, params):
    for o in domprob.ground_operator(op_name):
        if params == list(o.variable_list.values()) and o.precondition_pos.issubset(state_grounded_atoms):
            next_state_grounded_atoms = state_grounded_atoms
            for effect in o.effect_pos:
                next_state_grounded_atoms.append(effect)
            for effect in o.effect_neg:
                next_state_grounded_atoms.remove(effect)
            return next_state_grounded_atoms
    return None

def apply_grounded_plan(state_grounded_atoms, plan):
    plan_grounded_atoms = []
    plan_grounded_atoms.append(state_grounded_atoms)
    for ground_operator in plan:
        op_name = ground_operator[0]
        params = list([ground_operator[1]] if len(ground_operator[1]) == 2 else ground_operator[1:])
        plan_grounded_atoms.append(apply_grounded_operator(plan_grounded_atoms[-1], op_name, params))
        print("pga:", plan_grounded_atoms[-1])
        # TODO (wmcclinton) Fix
    return plan_grounded_atoms



