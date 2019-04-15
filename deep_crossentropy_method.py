from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import gym

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
    """

    __hidden_layer_size=(20,20)
    __nn_max_iter=1

    def __init__(self,_env):
        self.env=_env
        self.agent=MLPClassifier(hidden_layer_sizes=self.__hidden_layer_size,
        activation="tanh",warm_start=True,max_iter=self.__nn_max_iter)
        self.n_actions=self.env.action_space.n
        self.agent.fit([self.env.reset()]*self.n_actions,list(range(self.n_actions)))
        self.__fig=plt.figure(figsize=[8,4])
        self.__ax1= self.__fig.add_subplot(111)
        self.__ax2= self.__fig.add_subplot(122)
        self.__fig.show()


    def __generate_sessions(self,t_max=1000):
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

            a=np.random.choice(a=len(probs),size=1,p=probs)[0]

            new_s,r,done,info=self.env.step(a)

            states.append(s)
            actions.append(a)
            total_reward+=r

            s=new_s
            if done:break
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

        elite_states=[j for i in range(len(rewards_batch)) if rewards_batch[i]>=reward_threshold for j in states_batch[i]]
        elite_actions=[j for i in range(len(rewards_batch)) if rewards_batch[i]>=reward_threshold for j in actions_batch[i]]

        return elite_states,elite_actions

    def __show_progress(self,rewards_batch,log,percentile,reward_range=[-990,+10]):
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

        self.__ax1.clear()
        self.__ax1.plot(list(zip(*log))[0],label="Mean rewards")
        self.__ax1.plot(list(zip(*log))[1],label="rewards threshold")
        self.__ax1.legend()
        self.__ax1.grid()
        
        self.__ax2.clear()
        self.__ax2.hist(rewards_batch,reward_range)
        self.__ax2.vlines(np.percentile(rewards_batch,percentile),[0],[100],label="percentile",color="red")
        self.__ax2.legend()
        self.__ax2.grid()

        self.__fig.canvas.draw()


    def fit(self,n_sessions=100,percentile=50,iter=100,show=False):
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

            sessions=[self.__generate_sessions() for i in range(n_sessions)]

            states_batch,actions_batch,rewards_batch=map(np.array,zip(*sessions))

            elite_states,elite_actions=self.__select_elites(states_batch,actions_batch,rewards_batch,percentile)

            self.agent.fit(elite_states,elite_actions)

            if show:
                self.__show_progress(rewards_batch,log,percentile,reward_range=[0,np.max(rewards_batch)])

    def predict(self,states):
        probs=self.agent.predict_proba(states)

        actions=[]
        for i in probs:
            a=np.random.choice(a=len(probs),size=1,p=probs)[0]

            actions.append(a)

        return actions



dc=DeepCrossentropy(gym.make("CartPole-v0").env)

dc.fit(show=True)