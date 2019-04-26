import gym
from PlayAndTrain import play_and_train 
from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
import QLearningAgent as QL
import os
import time
env = gym.make("Taxi-v2")

n_actions = env.action_space.n
agent=QL.load("Taxi_v2_model")
# agent = QL.QLearningAgent(alpha=0.5, epsilon=0.25, discount=0.99,
#                        get_legal_actions = lambda s: range(n_actions))
s=env.reset()


while(True):
    a=agent.get_action(s)
    new_s,r,done,info=env.step(a)

    s=new_s
    os.system('clear')
    env.render()
    time.sleep(1)
    if(done):
        env.reset()
