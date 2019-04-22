import numpy as np 
import random 
from collections import defaultdict, OrderedDict, namedtuple
from environment import Environment
import itertools

Transition = namedtuple('Transition',['s','a','r','s_next','done'])

class QLearning(object):
    def __init__(self, config):
        self.Q_value = defaultdict(float)
        self.env = Environment(config=config)
        self.config = config
        self.reward_averaged = []
        self.reward_history = []

    def act(self, state):
        """
        Pick the best action according to Q values = argmax_a Q(s,a)
        Exploration is forced by epsilon-greedy
        """
        self.state_info, actions= self.env.generatePossibleAction(state)
        # import pdb; pdb.set_trace()
        # print(actions)
        if self.eps > 0. and np.random.rand() < self.eps:
            # select the action randomly
            return random.choice(actions)
        # import pdb; pdb.set_trace()
        qvals = {action: self.Q_value[self.state_info, action] for action in actions}
        max_q = max(qvals.values())

        # in case of multiple actions having the same Q values
        actions_with_max_q = [a for a,q in qvals.items() if q == max_q]
        return random.choice(actions_with_max_q)

    def _update_q_values(self, tr):
        # since the action space is changing every time we must adhere to that too 
        # now the new state or tr.s_next has possible actions which should be retrieved
        # print("Current state", tr.s)
        # print("future_state", tr.s_next)
        self.new_state_info, actions = self.env.generatePossibleAction(tr.s_next)
        # import pdb; pdb.set_trace()
        if actions == []:
            # print(actions)
            actions = [(-1,-1)]
        max_q_next = max([self.Q_value[self.new_state_info, a] for a in actions])
        # we do not include the values of the next state if terminated
        self.Q_value[self.state_info, tr.a] += self.config['ALPHA'] * (tr.r + self.config['GAMMA'] * max_q_next * (1-tr.done) - self.Q_value[self.state_info, tr.a])
        return
    
    def train(self):
        self.eps = self.config['EPSILON']
        for episode in range(self.config['N_EPISODES']):
            self.obs = self.env.reset()
            # import pdb; pdb.set_trace()   
            step = 0
            done = False
            reward_collected = 0
            while not done:
                # print(episode)
                action_to_take = self.act(self.obs)
                # print(action_to_take)
                # import pdb; pdb.set_trace()
                self.new_obs, thrput, done, info = self.env.step(action_to_take)
                reward = thrput * self.config['INIT_REWARD']
                if done:
                    if reward >= self.config['DESIRED_THROUGHPUT']:
                        reward = self.config['DONE_REWARD']
                    else:
                        reward = self.config['DONE_REWARD'] * -1
                self._update_q_values(Transition(self.obs, action_to_take, reward, self.new_obs, done))
                self.obs = self.new_obs
                step+=1
                reward_collected+=reward
            self.reward_history.append(reward_collected)
            self.reward_averaged.append(np.average(self.reward_history[-10:]))
            with open('logfile_25_100.txt','a') as fp:
                fp.write("Episode {}\nReward {}\nThroughPut {}\n".format(str(episode), str(reward_collected), str(thrput)))
        print("Training Over")
        # print(self.env.config['RELEASE_ACTION'])
        return

if __name__ == '__main__':
    config = {
		'NUM_MACHINES' : 25,
		'NUM_JOBS' : 50,
		'SCHEDULE_ACTION' : [],
		'RELEASE_ACTION' : [],
		'SIMULATION_TIME' : 25000,
		'ALPHA' : 0.01,
		'EPSILON' : 0.7,
		'N_EPISODES' : 100,
		'GAMMA' : 0.9,
		'INIT_REWARD' : 3,
		'DONE_REWARD' : 10,
		'DESIRED_THROUGHPUT' : .5}
    order = []
    for i in range(config['NUM_JOBS']):
        mac = [i for i in range(config['NUM_MACHINES'])]
        mac = random.sample(mac, config['NUM_MACHINES'])
        order.append(tuple(mac))
    config['ORDER_OF_PROCESSING'] = order
    model = QLearning(config=config)
    model.train()

