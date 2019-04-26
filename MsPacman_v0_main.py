import QLearningAgent as QL
import numpy as np
import gym
import matplotlib.pyplot as plt
from PlayAndTrain import play_and_train

env=gym.make("MsPacman-v0")

n_actions=env.action_space.n

agent = QL.QLearningAgent(alpha=0.5,epsilon=0.25,discount=0.99
,get_legal_actions=lambda s:range(n_actions))

size=np.array(env.observation_space.high).shape

states=[]

s=env.reset()

states.append(s)

for i in range(10000):

    a=np.random.choice(n_actions)
    new_s,r,done,info=env.step(a)

    states.append(new_s)
    s=new_s
    if done:
        s=env.reset()
    
for i in range(size[0]):
    for j in range(size[1]):
        for k in range(size[2]):
            t=states[:][i][j][k]
            plt.hist(t,bins=100,range=(0,255))
            plt.ion()
            plt.pause(0.2)
            plt.close()
