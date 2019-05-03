import numpy as np
import matplotlib.pyplot as plt
import QLearningAgent as QL
import PlayAndTrain as pat
import gym
import tensorflow as tf
import keras
import keras.layers as L

class ApproximateQLearningAgent:

    def __init__(self,env, alpha, epsilon,gamma,state_dim,n_action):
        self.alpha=alpha
        self.env=env
        self.epsilon=epsilon
        self.gamma=gamma
        self.state_dim=state_dim
        self.n_actions=n_actions
        with tf.device("/device:GPU:0"):
            config=tf.ConfigProto()
            config.gpu_options.allow_growth=True
            config.gpu_options.per_process_gpu_memory_fraction=0.2
            self.__sess=tf.Session(config=config)
            keras.backend.set_session(self.__sess)
            self.__network=keras.models.Sequential()
            self.__network.add(L.InputLayer(state_dim))
            self.__network.add(L.Dense(50,activation="relu"))
            self.__network.add(L.Dense(50,activation="relu"))
            self.__network.add(L.Dense(self.n_actions))
            self.__states_ph=keras.backend.placeholder(dtype="float32",shape=(None,)+self.state_dim)
            self.__actions_ph=keras.backend.placeholder(dtype="int32",shape=(None)+self.n_actions)
            self.__rewards_ph = keras.backend.placeholder(dtype='float32', shape=[None])
            self.__next_states_ph = keras.backend.placeholder(dtype='float32', shape=(None,) + self.state_dim)
            self.__is_done_ph = keras.backend.placeholder(dtype='bool', shape=[None])
            #get q-values for all actions in current states
            self.__predicted_qvalues = self.__network(self.__states_ph)

            #select q-values for chosen actions
            self.__predicted_qvalues_for_actions = tf.reduce_sum(self.__predicted_qvalues * tf.one_hot(self.__actions_ph, self.n_actions), axis=1)
            self.__predicted_next_qvalues = self.__network(self.__next_states_ph)

            # compute V*(next_states) using predicted next q-values
            self.__next_state_values = tf.reduce_max(self.__predicted_next_qvalues,axis=1)

            # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
            self.__target_qvalues_for_actions = self.__rewards_ph+self.gamma*self.__next_state_values

            # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
            self.__target_qvalues_for_actions = tf.where(self.__is_done_ph, self.__rewards_ph, self.__target_qvalues_for_actions)    

            self.__square_loss = (self.__predicted_qvalues_for_actions - tf.stop_gradient(self.__target_qvalues_for_actions)) ** 2
            self.__loss = tf.reduce_mean(self.__square_loss)
            self.__train_step = tf.train.AdamOptimizer(self.alpha).minimize(self.__loss)
        
    def get_action(self,state):
        """
        sample actions with epsilon-greedy policy
        recap: with p = epsilon pick random action, else pick action with highest Q(s,a)
        """
        
        q_values = self.__network.predict(state[None])[0]
        
        ###YOUR CODE
        if np.random.rand()<self.epsilon:
            return np.random.choice(self.n_actions)
        return np.argmax(q_values)
    
    def generate_session(self,t_max=1000):
        """play env with approximate q-learning agent and train it at the same time"""
        total_reward = 0
        s = self.env.reset()
        
        for t in range(t_max):
            a = self.get_action(s)       
            next_s, r, done, _ = self.env.step(a)
        
            self.__sess.run(self.__train_step,{
                self.__states_ph: [s], self.__actions_ph: [a], self.__rewards_ph: [r], 
                self.__next_states_ph: [next_s], self.__is_done_ph: [done]
            })

            total_reward += r
            s = next_s
            if done: break
                
        return total_reward
    def fit(self,decrease_rate=0.99,iter=1000,t_max=1000,session_iter=100):
        rewards=[]
        for i in range(iter):
            session_rewards = [self.generate_session(t_max=t_max) for _ in range(session_iter)]
            print("epoch #{}\tmean reward = {:.3f}\tepsilon = {:.3f}".format(i, np.mean(session_rewards), self.epsilon))
            self.epsilon*=decrease_rate
            rewards.append(np.mean(session_rewards))
        return rewards
    def predict(self,state):
        return self.get_action(state)

env=gym.make("BipedalWalker-v2")

print(env.observation_space.high)