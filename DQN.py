import math
from environment import Environment
import itertools
from collections import defaultdict, OrderedDict, namedtuple
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# torch.set_default_tensor_type(torch.FloatTensor)

Transition = namedtuple('Transition',['s','a','r','s_next','done'])

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class dqn_network(nn.Module):
        def __init__(self, n_mac, n_job):
            super(dqn_network, self).__init__()
            self.n_mac = n_mac 
            self.n_job = n_job 
            self.linear1 = nn.Linear(self.n_mac*self.n_job, self.n_mac)
        
        def forward(self, X):
            # assert something
            # assert X.dtype == torch.int32
            X = X.unsqueeze(0)
            # import pdb; pdb.set_trace()
            l1 = self.linear1(X)
            return l1  

class DQNLearner(object):
    def __init__(self, config):
        self.config = config 
        self.env = Environment(config=config)
        self.memory = ReplayMemory(10000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nmac = self.config['NUM_MACHINES']
        self.njob = self.config['NUM_JOBS']
        self.policy_net = dqn_network(self.nmac, self.njob)
        self.target_net = dqn_network(self.nmac, self.njob)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.reward_history = []
        self.reward_averaged = []

    def optimize_model(self):
        if len(self.memory) < self.config['BATCH_SIZE']:
            return
        transitions = self.memory.sample(self.config['BATCH_SIZE'])
        batch = Transition(*zip(*transitions))
        # import pdb; pdb.set_trace()
        s_next = torch.from_numpy(batch.s_next[0])
        s = torch.from_numpy(batch.s[0])
        r = torch.tensor([batch.r[0]])
        a = torch.tensor(batch.a[0])
        non_final_mask = torch.tensor(tuple(map(lambda s_:s_ is not None, s_next)), dtype=torch.uint8)
        # import pdb; pdb.set_trace()
        # non_final_next_states = s_next    # torch.cat([s_ for s_ in s_next if s_ is not None])
        # state_batch = torch.cat(s)
        # action_batch = torch.cat(a)
        # reward_batch = torch.cat(r)

        import pdb; pdb.set_trace()
        state_action_values = self.policy_net(s)
        
        next_state_values = torch.zeros(self.config['BATCH_SIZE'])
        next_state_values[non_final_mask] = self.target_net(s_next).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.config['GAMMA']) + r

        # print(state_action_values.shape)
        # print(expected_state_action_values.shape)

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1,1)
        self.optimizer.step()
        return
    
    def obsToState(self, obs):
        if obs is None:
            return None
        state = np.zeros((self.nmac*self.njob), dtype=np.float32)
        for i,o in enumerate(obs):
            # the jobs are saved in +1 format for ease of neurons
            o_list = [x+1 for x in list(o)]  + [0]*(self.njob-len(o))
            state[i*self.nmac:(i+1)*self.nmac] = o_list
        # print(state)
        return state

    def actAgent(self, obs):
        state_info, actions = self.env.generatePossibleAction(obs)
        self.state = self.obsToState(state_info)
        if self.config['EPSILON'] > 0. and np.random.rand() < self.config['EPSILON']:
            return random.choice(actions)
        else:
            with torch.no_grad():
                # import pdb; pdb.set_trace()
                action = self.policy_net(torch.from_numpy(self.state))
                action = action.ceil().squeeze().int().numpy()
                # print(action.shape)
                #-1 from every action
                return tuple(action-1)
    
    def transAction(self, action):
        action_ = list(action)
        # print(action_)
        for i,a in enumerate(action_):
            action_[i]+=1
        return tuple(action_)

    def train_policy(self):
        # eps = config['EPSILON']
        for episode in range(self.config['N_EPISODES']):
            obs = self.env.reset()
            done = False
            reward_collected = 0
            # obs_state = self.obsToState(obs)    # the state is represented in matrix of nmac*njob
            while not done:
                # take action
                action_to_take = self.actAgent(obs)

                # take a step
                print("Action to take",action_to_take)       # format : (0:2,1:0,2:-1,3:0,4:-1)
                import pdb; pdb.set_trace()
                new_obs, throughput, done, info = self.env.step(action_to_take)
                reward = throughput * self.config['INIT_REWARD']
                if done:
                    new_obs = None
                    if reward >= self.config['DESIRED_THROUGHPUT']:
                        reward = self.config['DONE_REWARD']
                    else:
                        reward = self.config['DONE_REWARD'] * -1
                else:
                    new_state_info,_ = self.env.generatePossibleAction(new_obs)
                    self.new_state = self.obsToState(new_state_info)

                # store to the memory
                self.memory.push(self.state, self.transAction(action_to_take), reward, self.new_state, done)

                obs = new_obs
                reward_collected+=reward

                # optimize the given model
                self.optimize_model()

            # if the a fixed taget update peroiod passes copy the dict values to target network
            self.reward_history.append(reward_collected)
            self.reward_averaged.append(np.average(self.reward_history[-10:]))
            with open('logfile_5_2000.txt','a') as fp:
                fp.write("Episode {}\nReward {}\nThroughPut {}\n".format(str(episode), str(reward_collected), str(thrput)))
        print("Training Over")


def main():
    # config parameters
    config = {
		'NUM_MACHINES' : 5,
		'NUM_JOBS' : 5,
		'ORDER_OF_PROCESSING' : [(0,2,3,1,4),(2,4,1,3,0),(3,1,2,4,0),(1,0,2,3,4),(3,0,4,2,1)],
		'SCHEDULE_ACTION' : [],
		'RELEASE_ACTION' : [],
		'SIMULATION_TIME' : 10000,
		'ALPHA' : 0.01,
		'EPSILON' : 0.5,
		'N_EPISODES' : 1,
		'GAMMA' : 0.9,
		'INIT_REWARD' : 3,
		'DONE_REWARD' : 10,
		'DESIRED_THROUGHPUT' : .25,
        'BATCH_SIZE' : 1,
        'TARGET_UPDATE' : 10
    }
    # training loop
    learner = DQNLearner(config=config)
    learner.train_policy()

if __name__ == "__main__":
    # config dictionary here
    main()



