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

step=0
while True:
    for i in range(50):
        env.render()
        # vid.capture_frame()
        action = np.asarray([0, 0, 0, 0])
        step_results = env.step(action)
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
