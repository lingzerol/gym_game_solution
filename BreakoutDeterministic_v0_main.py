import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces.box import Box
from gym.core import Wrapper
from gym.core import ObservationWrapper
from PIL import Image
import tensorflow as tf
from keras.layers import Conv2D,Dense,Flatten
import keras
from tqdm import trange
from pandas import DataFrame
import dill
import getopt,sys
class FrameBuffer(Wrapper):
    """
    memory the latest n_frames state of the environment
    """
    def __init__(self,env,n_frames=4):
        Wrapper.__init__(self,env)
        height,width,n_channels=env.observation_space.shape
        obs_shape=[height,width,n_channels*n_frames]
        self.env=env
        self.observation_space=Box(0.0,1.0,obs_shape)
        self.framebuffer=np.zeros(obs_shape,'float32')

    def reset(self):
        """
        reset the env
        """
        self.framebuffer=np.zeros_like(self.framebuffer)
        self.update_buffer(self.env.reset())
        return self.framebuffer

    def step(self,action):
        """
        taking action

        parameters:
        action - action the env should take

        returns:
        framebuffer - latest n_frame state
        reward - the reward of the action
        done - whether the game is finished or not
        info - ...
        """
        s,reward,done,info=self.env.step(action)
        self.update_buffer(s)
        return self.framebuffer,reward,done,info

    def update_buffer(self,state):
        """
        add new state to the framebuffer, and delete the last state

        parameters:
        state - new state of env

        returns:
        null
        """
        cropped_frambuffer=self.framebuffer[:,:,:-self.env.observation_space.shape[-1]]
        self.framebuffer=np.concatenate([state,cropped_frambuffer],axis=-1)


class ReplayBuffer:
    def __init__(self,size):
        """
        Create Replaybuffer - store the latest "size" states
        
        parameters:
        size - the number of state replaybuffer stores
        """

        self.size=size
        self._storage=[]
        self._next_id=0

    def __len__(self):
        """
        return the number of state stored
        """
        return len(self._storage)

    def add(self,state,action,reward,next_s,done):
        """
        add new pair of state, reward, next_s and done information
        
        parameters:
        state - the orginal state
        action - action the enviroment took
        reward - reward got after taking action
        next_s - state after taking action
        done - whether the game is done or not

        returns:
        """

        data=(state,action,reward,next_s,done)

        if(self._next_id>=len(self._storage)):
            self._storage.append(data)
        else:
            self._storage[self._next_id]=data
        self._next_id=(self._next_id+1)%self.size

    def _encode_sample(self,indexes):
        """
        get data with indexes
        parameters:
        indexes - list, contain the index of data the function should return

        returns:
        states - array like, states of the data
        actions - array like, actions of the data
        rewards - array like, rewards of the data
        next_ss - array like, new states of the data
        dones - array like, new dones of the data
        """
        states,actions,rewards,next_ss,dones=[],[],[],[],[]

        for i in indexes:
            data = self._storage[i]
            states.append(np.array(data[0],copy=False))
            actions.append(np.array(data[1],copy=False))
            rewards.append(data[2])
            next_ss.append(np.array(data[3],copy=False))
            dones.append(data[4])
        return np.array(states),np.array(actions),np.array(rewards),np.array(next_ss),np.array(dones)

    def sample(self,batch_size):
        """
        randomly choosing batch_size of old data

        parameters:
        batch_size - number data should be sampled

        returns:
        self._encode_sample
        """
        indexes=np.random.choice(a=len(self._storage),size=batch_size)
        return self._encode_sample(indexes)


class PreprocessAtari(ObservationWrapper):
    def __init__(self,env):
        """
        preprocess environment observation shape
        """
        ObservationWrapper.__init__(self,env)

        self.shape=(60,60)
        self.observation_space=Box(0.0,1.0,(self.shape[0],self.shape[1],1))

    def observation(self,state):
        """
        get the processed state
        
        parameters:
        state - origin state

        returns:
        state - processed state
        """

        state=state[34:-15][:][:]
        state=np.array(Image.fromarray(state).resize(self.shape))
        state=state.mean(axis=-1,keepdims=True)
        state=state.astype('float32')/255
        return state


def makeAtarienv(game:str,n_frames=4):
    env=gym.make(game)
    env=PreprocessAtari(env)
    env=FrameBuffer(env,n_frames=n_frames)
    return env

class DQNNetwork:
    """
    class implement deep q learning
    """
    def __init__(self,sess,name,state_shape,n_actions,epsilon=0,reuse=False):
        """
        parameters:
        name - variable namespace
        state_shape - shape of state
        n_actions - number of actions can be token
        epsilon - using greedy-epsilon strategy
        resuse - resuse the variable in the variable namespace
        """
        self.sess=sess
        with tf.variable_scope(name,reuse=reuse):
            #with tf.device("/cpu:0"):
            with tf.device("/device:GPU:0"):
                self.network=keras.models.Sequential()
                self.network.add(Conv2D(16,(3,3),strides=2,activation="relu",input_shape=state_shape))
                self.network.add(Conv2D(32,(3,3),strides=2,activation="relu"))
                self.network.add(Conv2D(64,(3,3),strides=2,activation="relu"))
                self.network.add(Flatten())
                self.network.add(Dense(256,activation="relu"))
                self.network.add(Dense(n_actions,activation="linear"))

                self.states=tf.placeholder('float32',[None,]+list(state_shape),name="states")
                #print(self.states)
                self.qvalues=self.get_symbolic_qvalues(self.states)
        self.sess.run(tf.global_variables_initializer())
        self.weights=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=name)
        self.epsilon=epsilon

    def get_symbolic_qvalues(self,states):
        """
        get qvalue tensor with the state
        
        parameters:
        states - a tensor, calculate the qvalues of the state

        returns:
        qvalues - a tensor, qvalues of state
        """
        qvalues = self.network(states)
        return qvalues

    def get_qvalues(self,states):
        return self.sess.run(self.qvalues,{self.states:states})

    def get_actions(self,qvalues):
        epsilon=self.epsilon
        batch_size,n_actions=qvalues.shape
        random_actions=np.random.choice(n_actions,size=batch_size)
        best_action=qvalues.argmax(axis=-1)
        should_explore=np.random.choice([0,1],batch_size,p=[1-epsilon,epsilon])
        return np.where(should_explore,random_actions,best_action)


class DQNAgent:
    def __init__(self,env,epsilon=0,gamma=0.99,boundary=500,decrease_epsilon=0.99,load_sess=False,file="",global_step=1,reuse=False):

        self.env=env
        self.state_shape=env.observation_space.shape
        self.n_actions=env.action_space.n
        self.epsilon=epsilon
        self.exp_replay=ReplayBuffer(10**5)


        #with tf.device("/cpu:0"):
        with tf.device("/device:GPU:0"):
            config=tf.ConfigProto()
            config.gpu_options.allow_growth=True
            config.gpu_options.per_process_gpu_memory_fraction=0.2
            self.sess=tf.Session(config=config)
            self.network=DQNNetwork(self.sess,"network",self.state_shape,self.n_actions,epsilon=epsilon,reuse=reuse)
            self.target_network=DQNNetwork(self.sess,"target_network",self.state_shape,self.n_actions,epsilon=epsilon,reuse=reuse)
            with tf.variable_scope("DQNAgent",reuse=reuse):
                self.states=tf.placeholder("float32",shape=(None,)+self.state_shape,name="states")
                self.actions=tf.placeholder("int32",shape=[None],name="actions")
                self.rewards=tf.placeholder("float32",shape=[None],name="rewards")
                self.next_ss=tf.placeholder("float32",shape=(None,)+self.state_shape,name="next_ss")
                self.done=tf.placeholder("float32",shape=[None],name="done")

                self.is_not_done=1-self.done
                self.gamma=gamma

                self.current_qvalues=self.network.get_symbolic_qvalues(self.states)
                self.current_action_qvalues=tf.reduce_sum(tf.one_hot(self.actions,self.n_actions)*self.current_qvalues,axis=1)

                self.next_qvalues_target=self.target_network.get_symbolic_qvalues(self.next_ss)
                self.next_state_values_target=tf.reduce_max(self.next_qvalues_target,axis=-1)

                self.reference_qvalues=self.rewards+self.gamma*self.next_state_values_target*self.is_not_done

                self.td_sub_loss=(self.current_action_qvalues-self.reference_qvalues)**2
                self.td_loss=tf.reduce_mean(self.td_sub_loss)

                self.train_step=tf.train.AdamOptimizer(1e-3).minimize(self.td_loss,var_list=self.network.weights)
        if load_sess==True:
            self.load(file,global_step)
        self.sess.run(tf.global_variables_initializer())

        self.times=0


    def __load_weights_into_target_network(self):
        assigns=[]
        for w_agent,w_target in zip(self.network.weights,self.target_network.weights):
            assigns.append(tf.assign(w_agent,w_target,validate_shape=True))
        self.sess.run(assigns)

    def __sample_batch(self,batch_size=64):
        states,actions,rewards,next_ss,dones=self.exp_replay.sample(batch_size)
        return {self.states:states,self.actions:actions,self.rewards:rewards,self.next_ss:next_ss,self.done:dones}


    def play_and_record(self,n_step):
        """
        Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer. 
        Whenever game ends, add record with done=True and reset the game.

        parameters:
        agent - a DQNAgent
        env - gym environment
        exp_replay - a ReplayBuffe, used to store the data
        n_step - number of steps
        returns:
        return sum of rewards over time
        """

        s=self.env.framebuffer

        reward=0.0
        for i in range(n_step):
            action=self.predict([s])[0]
            next_s,r,done,_=self.env.step(action)

            self.exp_replay.add(s,action,r,next_s,done)
            reward+=r
            if done:
                s=self.env.reset()
            else:
                s=next_s
        return reward

    def fit(self,t_max=10**5):
        if(len(self.exp_replay)<self.exp_replay.size):
            self.play_and_record(self.exp_replay.size)
        mean_rw_history = []
        td_loss_history = []
        
        for i in range(t_max):

            # play
            self.play_and_record(10)

            # train
            _, loss_t = self.sess.run([self.train_step, self.td_loss], self.__sample_batch(batch_size=64))
            td_loss_history.append(loss_t)
            
            # adjust agent parameters
            if i % 500 == 0:
                self.__load_weights_into_target_network()
                self.network.epsilon = max(self.network.epsilon * 0.99, 0.01)
                mean_rw_history.append(self.evaluate(n_games=3))
                if i % 100 == 0:
                    print("index %i"%(i),end=":\n")
                    print("buffer size = %i, epsilon = %.5f" % (len(self.exp_replay), self.network.epsilon),end=",")
                    print(" mean_reward:%0.5f, td_loss:%0.5f"%(mean_rw_history[-1],td_loss_history[-1]))
                    self.save("./BreakoutDeterministic/model/model.ckpt",int(i/100))
                with open("./BreakoutDeterministic/history/mean_rw_history", "a+") as f:
                    for i in mean_rw_history:
                        f.write(str(i)+" ")
                    f.write("\n")
                    f.close()
                    mean_rw_history.clear()
                with open("./BreakoutDeterministic/history/td_loss_history", "a+") as f:
                    for i in td_loss_history:
                        f.write(str(i)+" ")
                    f.write("\n")
                    f.close()
                    td_loss_history.clear()
 
        return mean_rw_history,td_loss_history


    def predict(self,state):
        qvalues=self.network.get_qvalues(state)
        action=self.network.get_actions(qvalues)
        return action

    def load(self,file:str,global_step=1):
        saver=tf.train.Saver()
        saver.restore(self.sess,file+"-"+str(global_step))
    def save(self,file:str,global_step=1):
        saver=tf.train.Saver()
        saver.save(self.sess,file,global_step=global_step)

    def evaluate(self,n_games=1, greedy=False, t_max=10000):
        """ 
        evaluate the agent
        Plays n_games full games. If greedy, picks actions as argmax(qvalues)..
        
        parameters:
        env - gym environment
        agent - a DQNAgent
        n_games - the number of games which will be played
        greedy - using epsilon-greedy strategy or not
        t_max - the number of iterators in a game

        returns:
        mean_reward
        """
        rewards = []
        for _ in range(n_games):
            s = self.env.reset()
            reward = 0
            for _ in range(t_max):
                qvalues = self.network.get_qvalues([s])
                action = qvalues.argmax(axis=-1)[0] if greedy else self.network.get_actions(qvalues)[0]
                s, r, done, _ = self.env.step(action)
                reward += r
                if done: break
                    
            rewards.append(reward)
        return np.mean(rewards)




def show(mean_rw_history,td_loss_history):
    moving_average = lambda x, span, **kw: DataFrame({'x':np.asarray(x)}).x.ewm(span=span, **kw).mean().values
    plt.figure(figsize=[12,4])
    plt.subplot(1,2,1)
    plt.title("mean reward per game")
    plt.plot(mean_rw_history)
    plt.grid()

    #plt.figure(figsize=[12, 4])
    plt.subplot(1,2,2)
    plt.title("TD loss history (moving average)")
    plt.plot(moving_average(np.array(td_loss_history), span=100, min_periods=100))
    plt.grid()
    plt.show()


def main():
    file=""
    global_step=0
    if(len(sys.argv)>1):
        try:
            options,argsp=getopt.getopt(sys.argv[1:],"f:g:",["file=","global_step="])
        except getopt.GetoptError:
            sys.exit()
        for option,value in options:
            if option in ("-f","--file"):
                file=value
            if option in ("-g","--global_step"):
                global_step=int(value)
    print(file,global_step)
    env=makeAtarienv("BreakoutDeterministic-v4")
    env.reset()
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape

    load_sess=False
    if file!="" and global_step!=0:
        load_sess=True
    agent=DQNAgent(env,epsilon=0.5,load_sess=load_sess,file=file,global_step=global_step)
    mean_rw_history,td_loss_history=agent.fit()

    import gym.wrappers
    env.reset()
    env_monitor = gym.wrappers.Monitor(env,directory="./BreakoutDeterministic/videos",force=True)
    sessions = [agent.evaluate(n_games=1) for _ in range(100)]
    env_monitor.close()


if __name__=='__main__':
    main()