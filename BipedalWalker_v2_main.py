import numpy as np
import matplotlib.pyplot as plt
import QLearningAgent as QL
import PlayAndTrain as pat
import gym

env=gym.make("BipedalWalker-v2")
print(env.action_space.low)
n_actions=env.action_space.n
states=[]
s=env.reset()
states.append(s)
for i in range(1):
    a=np.random.choice(n_actions)
    new_s,r,done,info=env.step(a)

    states.append(new_s)
    s=new_s
    if done:
        s=env.reset()

for i in range(env.observation_space.n):
    plt.hist(states[:][i],bins=50)
    plt.show()
