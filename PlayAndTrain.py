import random
from QLearningAgent import ReplayBuffer
def play_and_train(env,agent,t_max=10**4):
    """
    This function should 
    - run a full game, actions given by agent's e-greedy policy
    - train agent using agent.update(...) whenever it is possible
    - return total reward
    """
    total_reward = 0.0
    s = env.reset()
    
    for t in range(t_max):
        # get agent to pick action given state s.
        a = agent.get_action(s)
        
        next_s, r, done, _ = env.step(a)
        
        # train (update) agent for state s
        agent.update(s,a,r,next_s)
        
        s = next_s
        total_reward +=r
        if done: break
        
    return total_reward

def play_and_train_with_replay(env, agent, replay=None, 
                               t_max=10**4, replay_batch_size=32):
    """
    This function should 
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total reward
    :param replay: ReplayBuffer where agent can store and sample (s,a,r,s',done) tuples.
        If None, do not use experience replay
    """
    total_reward = 0.0
    s = env.reset()
    
    for t in range(t_max):
        # get agent to pick action given state s
        a = agent.get_action(s)
        
        next_s, r, done, _ = env.step(a)

        # update agent on current transition. Use agent.update
        agent.update(s,a,r,next_s)
        

        if replay is not None:
            # store current <s,a,r,s'> transition in buffer
            replay.add(s,a,r,next_s,done)
            
            # sample replay_batch_size random transitions from replay, 
            # then update agent on each of them in a loop
            for state,action,reward,next_state,is_done in zip(*list(replay.sample(replay_batch_size))):
                if is_done:
                    break
                agent.update(state,action,reward,next_state)
            
                    
        s = next_s
        total_reward +=r
        if done:break
    
    return total_reward