from deep_crossentropy_method import DeepCrossentropy
from sklearn.neural_network import MLPClassifier
import numpy as np
import gym
import matplotlib.pyplot as plt
import QLearningAgent as QL
import PlayAndTrain as pat
from pandas import DataFrame


def DC_CartPole_v1():
    env = gym.make("CartPole-v0").env

    agent = MLPClassifier(hidden_layer_sizes=(
        20, 20), activation="tanh", warm_start=True, max_iter=1)

    dc = DeepCrossentropy(env, agent, list(range(env.action_space.n)))

    dc.load("CartPole_v1_model.sav")

    s = env.reset()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    while(True):
        a = dc.predict([s])[0]
        new_s, r, done, info = env.step(a)

        s = new_s
        ax.clear()
        ax.imshow(env.render('rgb_array'))
        fig.canvas.show()
        if(done):
            env.reset()


def QL_CartPole_v1():
    class CartPoleBinarizer(QL.Binarizer):
        def _observation(self, state):
            n_digits = [0, 1, 2, 1]
            for i in range(len(state)):
                state[i] = np.round(state[i], n_digits[i])

            return tuple(state)

    env = CartPoleBinarizer(gym.make("CartPole-v1"))
    n_actions = env.action_space.n
    agent = QL.EVSarsaAgent(alpha=0.5, epsilon=0.25, discount=0.99,
                            get_legal_actions=lambda s: range(n_actions))

    def moving_average(x, span=100): return DataFrame({'x': np.asarray(x)}).x.ewm(span=span).mean().values
    sarsa_rewards = []

    for i in range(5000):
        sarsa_rewards.append(pat.play_and_train(env, agent))
       
        if(i % 100==0):
            print('EVSARSA mean reward =', np.mean(sarsa_rewards[-100:]))
            plt.title("epsilon = %s" % agent.epsilon)
            plt.plot(moving_average(sarsa_rewards), label='ev_sarsa')
            plt.grid()
            plt.legend()
            plt.ion()
            plt.pause(1)
            plt.close()
    QL.save(agent,"CartPole_v1_sarsa_model")
    s = env.reset()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    while(True):
        a = agent.get_best_action(s)
        new_s, r, done, info = env.step(a)

        s = new_s
        ax.clear()
        ax.imshow(env.render('rgb_array'))
        fig.canvas.show()
        if(done):
            env.reset()


def AQL_CartPole_v1():
    env = gym.make("CartPole-v0").env
    agent=QL.ApproximateQLearningAgent(env=env,alpha=1e-4,epsilon=0.5,gamma=0.99,n_actions=env.action_space.n,state_dim=env.observation_space.shape)

    rewards=agent.fit(iter=100)
    print(rewards)
