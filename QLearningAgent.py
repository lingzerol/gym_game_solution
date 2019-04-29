import numpy as np
import gym
from collections import defaultdict
import pickle
from gym.core import ObservationWrapper
import dill
import random
import tensorflow as tf
import keras
import keras.layers as L
class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Q-Learning Agent
        based on http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
        Instance variables you have access to
          - self.epsilon (exploration prob)
          - self.alpha (learning rate)
          - self.discount (discount rate aka gamma)

        Functions you should use
          - self.get_legal_actions(state) {state, hashable -> list of actions, each is hashable}
            which returns legal actions for a state
          - self.get_qvalue(state,action)
            which returns Q(state,action)
          - self.set_qvalue(state,action,value)
            which sets Q(state,action) := value

        !!!Important!!!
        Note: please avoid using self._qValues directly. 
            There's a special self.get_qvalue/set_qvalue for that.
        """

        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvalues[state][action]

    def set_qvalue(self,state,action,value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[state][action] = value

    #---------------------START OF YOUR CODE---------------------#

    def get_value(self, state):
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        Note: please take into account that q-values can be negative.
        """
        possible_actions = self.get_legal_actions(state)

        #If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        value=max(self.get_qvalue(state,action) for action in possible_actions)

        return value

    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        #agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        new_q=(1-learning_rate)*self.get_qvalue(state,action)+learning_rate*(reward+gamma*self.get_value(next_state))
        
        self.set_qvalue(state, action, new_q)

    
    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values). 
        """
        possible_actions = self.get_legal_actions(state)

        #If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        q_values={action:self.get_qvalue(state,action) for action in possible_actions}
        best_action=max(q_values,key=q_values.get)

        return best_action

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.  
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.getPolicy).
        
        Note: To pick randomly from a list, use random.choice(list). 
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)
        action = None

        #If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        #agent parameters:
        epsilon = self.epsilon
        
        chosen=np.random.choice(a=2,size=1,p=[1-epsilon,epsilon])[0]
        
        if 0==chosen:
            chosen_action=self.get_best_action(state)
        else:
            chosen_action=possible_actions[np.random.choice(a=len(possible_actions),size=1)[0]]
        return chosen_action

class EVSarsaAgent(QLearningAgent):
    """ 
    An agent that changes some of q-learning functions to implement Expected Value SARSA. 
    Note: this demo assumes that your implementation of QLearningAgent.update uses get_value(next_state).
    If it doesn't, please add
        def update(self, state, action, reward, next_state):
            and implement it for Expected Value SARSA's V(s')
    """
    
    def get_value(self, state):
        """ 
        Returns Vpi for current state under epsilon-greedy policy:
          V_{pi}(s) = sum _{over a_i} {pi(a_i | s) * Q(s, a_i)}
          
        Hint: all other methods from QLearningAgent are still accessible.
        """
        epsilon = self.epsilon
        possible_actions = self.get_legal_actions(state)

        #If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        
        pi={action:epsilon/len(possible_actions) for action in possible_actions}
        pi[self.get_best_action(state)]+=1-epsilon
        state_value=sum(pi[action]*self.get_qvalue(state,action) for action in possible_actions)
        
        return state_value


class Binarizer(ObservationWrapper):
    
    def _observation(self, state):
        pass



class ReplayBuffer(object):
    def __init__(self, size):
        """
        Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
            
        Note: for this assignment you can pick any data structure you want.
              If you want to keep it simple, you can store a list of tuples of (s, a, r, s') in self._storage
              However you may find out there are faster and/or more memory-efficient ways to do so.
        """
        self._storage = []
        self._maxsize = size
        
        # OPTIONAL: YOUR CODE
        

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        '''
        Make sure, _storage will not exceed _maxsize. 
        Make sure, FIFO rule is being followed: the oldest examples has to be removed earlier
        '''
        data = (obs_t, action, reward, obs_tp1, done)
        
        # add data to storage
        if(self.__len__()>=self._maxsize):
            del self._storage[0]
        self._storage.append(data)
        
    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        
        idxes = random.sample(range(self.__len__()),min(self.__len__(),batch_size))
        # collect <s,a,r,s',done> for each index
        states,actions,rewards,next_states,is_done=[],[],[],[],[]
        for i in idxes:
            states.append(self._storage[i][0])
            actions.append(self._storage[i][1])
            rewards.append(self._storage[i][2])
            next_states.append(self._storage[i][3])
            is_done.append(self._storage[i][4])

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(is_done)
      

class ApproximateQLearningAgent:

    def __init__(self,env, alpha, epsilon,gamma,state_dim,n_actions):
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
            self.__network.add(L.Dense(n_actions))
            self.__states_ph=keras.backend.placeholder(dtype="float32",shape=(None,)+self.state_dim)
            self.__actions_ph=keras.backend.placeholder(dtype="int32",shape=[None])
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

def load(file:str):
    with open(file,"rb") as f:
        return dill.load(f)

def save(agent:QLearningAgent,file:str):
    with open(file, "wb") as f:
        dill.dump(agent,f)


    