
from env import ArmEnv
from rl import DDPG
import time
import numpy as np
MAX_EPISODES = 50000
MAX_EP_STEPS = 200
ON_TRAIN = True

# set env
env = ArmEnv()

# set RL method (continuous)
rl = DDPG()

steps = []

def train():
    # start training
    RENDER = False
    done_cnt = 0
    var = 2.00
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            if len(rl.memory) <= 9999:
                a = env.sample_action()
            else :
                a = rl.choose_action(s)
            a[0] = np.clip(np.random.normal(a[0], var), -1, 1)
            a[1] = np.clip(np.random.normal(a[1], var), -1, 1)

            s_, r, done = env.step(a)

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if len(rl.memory) > 9999:
                # start to learn once has fulfilled the memory
                rl.learn()
                var *= 0.9999
                # if var > 0.1:
                #     var *= 0.99995
            s = s_
            if done:
                done_cnt += 1
            if done_cnt >= 100 or i > 100:
                RENDER = True
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i | var = %.5f' % (i, '---' if not done else 'done', ep_r, j, var))
                break
            if i % 10 == 0 and i > 50 and RENDER:
                time.sleep(0.01)
    rl.save()






def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    while True:
        s = env.reset()
        for _ in range(300):
            env.render()
            a = rl.choose_action(s)
            s, r, done = env.step(a)
            if done:
                break


if ON_TRAIN:
    train()
else:
    eval()
