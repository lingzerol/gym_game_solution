from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import gym
from sklearn.externals import joblib
class DeepCrossentropy:
    """
    a class using deep crossentropy method to solve the gym game env

    parameters:
    public:
    env - the gym game
    agent - MLPClassifier instance, used to get the actions from different states
    n_actions - the size of actions space

    private:
    __hidden_layer_size - the hidden layer size of the agent
    __nn_max_iter -nerual network iteration times
    __fid - used to show the progress
    __ax1 - used to show the figure of mean reward and reward threshold
    __ax2 - used to show the hist of the rewards batch
    """

    

    def __init__(self,env,agent,actions_space):
        self.env=env
        self.agent=agent
        self.actions_space=actions_space
        




    def __generate_sessions(self,t_max=1000,actions_times=5):
        """
        generate session about the game
        
        parameters:
        t_max - the max iteration number

        returns:
        states - array-like, the states of all iteration
        actions - array-like, the actions token in each iteration
        total_reward - a number, the sum of the rewards got from iterations   
        """

        states,actions=[],[]
        total_reward=0

        s=self.env.reset()

        for i in range(t_max):

            probs=self.agent.predict_proba([s])[0]

            a=self.actions_space[np.random.choice(a=len(probs),size=1,p=probs)[0]]

            for j in range(actions_times):
                new_s,r,done,info=self.env.step(a)
                total_reward+=r
                if done:break

            states.append(s)
            actions.append(a)
            
            s=new_s
            if done:
                break
        return states,actions,total_reward

    def __select_elites(self,states_batch,actions_batch,rewards_batch,percentile=50):
        """
        Select states and actions from games that have rewards >= percentile

        parameters:
        states_batch: list of lists of states, states_batch[session_i][t]
        actions_batch: list of lists of actions, actions_batch[session_i][t]
        rewards_batch: list of rewards, rewards_batch[session_i][t]

        returns:
        elite_states: list of lists of states which have rewards >= percentile
        elite_actions: list of lists of actions which have rewards >= percentile
        """

        reward_threshold=np.percentile(rewards_batch,percentile)

        if np.max(rewards_batch)>reward_threshold:
            elite_states=[j for i in range(len(rewards_batch)) if rewards_batch[i]>reward_threshold for j in states_batch[i]]
            elite_actions=[j for i in range(len(rewards_batch)) if rewards_batch[i]>reward_threshold for j in actions_batch[i]]
        else:
            elite_states=[j for i in range(len(rewards_batch)) if rewards_batch[i]>=reward_threshold for j in states_batch[i]]
            elite_actions=[j for i in range(len(rewards_batch)) if rewards_batch[i]>=reward_threshold for j in actions_batch[i]]


        return elite_states,elite_actions

    def __show_progress(self,rewards_batch,log,percentile,reward_range=[-990,+10],show_progress=False):
        """
        A convenience function that displays training progress. 
        
        parameters:
        rewards_batch - list of rewards
        log - log the mean reward and reward threshold
        reward_range - tell the reward value range
        
        returns:
        
        """
     

        mean_reward,reward_threshold=np.mean(rewards_batch),np.percentile(rewards_batch,percentile)
        log.append([mean_reward,reward_threshold])

        print("mean reward = %.3f, reward threshold = %.3f"%(mean_reward,reward_threshold))

        if show_progress:
            plt.figure(figsize=[8,4])
            plt.subplot(1,2,1)
            plt.plot(list(zip(*log))[0],label="Mean rewards")
            plt.plot(list(zip(*log))[1],label="rewards threshold")
            plt.legend()
            plt.grid()
        
            plt.subplot(1,2,2)
            plt.hist(rewards_batch,range=reward_range)
            plt.vlines(np.percentile(rewards_batch,percentile),[0],[100],label="percentile",color="red")
            plt.legend()
            plt.grid()

            plt.ion()
            plt.pause(2)
            plt.close()


    def fit(self,n_sessions=100,percentile=50,iter=100,t_max=1000,actions_times=5,show_progress=False):
        """
        training the model

        parameters:
        n_session - number of session in each iteration
        percentile - percentage of elite states and actions
        iter - iterations' number

        return:
        """

        log=[]

        for i in range(iter):

            sessions=[self.__generate_sessions(t_max=t_max,actions_times=actions_times) for i in range(n_sessions)]

            states_batch,actions_batch,rewards_batch=map(np.array,zip(*sessions))

            elite_states,elite_actions=self.__select_elites(states_batch,actions_batch,rewards_batch,percentile)

            self.agent.fit(elite_states,elite_actions)

        
            self.__show_progress(rewards_batch,log,percentile,reward_range=[min(0,np.min(rewards_batch)),np.max(rewards_batch)],show_progress=show_progress)

    def predict(self,states):
        probs=self.agent.predict_proba(states)

        actions=[]
        for i in probs:
            a=self.actions_space[np.random.choice(a=len(i),size=1,p=i)[0]]

            actions.append(a)

        return actions

    def save(self,file:str):
        joblib.dump(self.agent,file)
    def load(self,file:str):
        self.agent=joblib.load(file)


env=gym.make("MountainCar-v0").env
n_actions=env.action_space.n
agent=MLPClassifier(hidden_layer_sizes=(20,30,50,30,20),
        activation="tanh",warm_start=True,max_iter=5)
agent.fit([env.reset()]*2,[0,2])

dc=DeepCrossentropy(env,agent,[0,2])
# dc.load("MountainCar_v0_model.sav")
at=[10,8,5,3,1]
pt=[80,70,60,50,50]

# for i,j in zip(at,pt):
#     dc.fit(iter=200,show_progress=False,percentile=j,actions_times=i,t_max=5000,n_sessions=100)
# dc.save("MountainCar_v0_model.sav")
dc.load("MountainCar_v0_model.sav")
s=env.reset()

fig=plt.figure()
ax=fig.add_subplot(111)
fig.show()

while True:
    a=dc.predict([s])[0]

    new_s,r,done,info=env.step(a)

    s=new_s
    ax.clear()
    ax.imshow(env.render("rgb_array"))
    fig.canvas.show()
