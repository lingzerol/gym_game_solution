from deep_crossentropy_method import DeepCrossentropy
from sklearn.neural_network import MLPClassifier
import numpy as np
import gym
import matplotlib.pyplot as plt

env=gym.make("CartPole-v0").env

agent=MLPClassifier(hidden_layer_sizes=(20,20),activation="tanh",warm_start=True,max_iter=1)

dc=DeepCrossentropy(env,agent,list(range(env.action_space.n)))

dc.load("CartPole_v1_model.sav")

s=env.reset()
fig=plt.figure()
ax=fig.add_subplot(111)

while(True):
    a=dc.predict([s])[0]
    new_s,r,done,info=env.step(a)

    s=new_s
    ax.clear()
    ax.imshow(env.render('rgb_array'))
    fig.canvas.show()
    if(done):
        env.reset()

