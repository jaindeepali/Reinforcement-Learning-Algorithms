import numpy as np
import collections
Step = collections.namedtuple("Step", ["observation", "reward", "done"])

class ToyEnv():
    def __init__(self):
        self.n_states = 5 # Number of states
        self.n_actions = 4 # Number of actions
        self.term_states = [self.n_states-2,self.n_states-1] # Terminal states        
        
        self.rewards = np.zeros([self.n_states, self.n_actions]) # Rewards
        self.rewards[-1] = 1 # Goal state
        self.rewards[-2] = -1 # Penalty state
        
        self.transition_prob = np.random.random([self.n_states,self.n_actions,self.n_states]) # Transition Probabilities
        s = self.transition_prob.sum(axis=-1)
        self.transition_prob = self.transition_prob/np.repeat(s, self.n_states).reshape([self.n_states, self.n_actions, self.n_states]) # Normalization
        self.transition_prob[-1] = 0 # Make goal state terminal
        self.transition_prob[-1,:,-1] = 1 # Make goal state terminal
        self.transition_prob[-2] = 0 # Make penalty state terminal
        self.transition_prob[-2,:,-2] = 1 # Make penalty state terminal
    
    @property
    def observation_space(self):
        return np.arange(self.n_states)
    
    @property
    def action_space(self):
        return np.arange(self.n_actions)    
    
    def reset(self):
        self._state = 0
        reward = self.rewards[self._state, 0]
        self.done = self._state in self.term_states
        return Step(observation=self._state, reward=reward, done=self.done)

    def step(self, action=None):
        reward = self.rewards[self._state, action]
        self.done = self._state in self.term_states
        self._state = np.random.choice(self.n_states, 1, p=self.transition_prob[self._state, action])[0]
        return Step(observation=self._state, reward=reward, done=self.done)

    def render(self):
        print('current state:', self._state)