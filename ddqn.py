#!/usr/bin/env/python3
# -*- coding: utf-8 -*-
'''DLP DDQN Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
from collections import deque, namedtuple
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
import numpy as np
import argparse
import itertools
import random
import time
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayMemory:
    #__slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def __len__(self):
        return len(self.buffer)
    def append(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self, batch_size, device):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.buffer, k=batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)


    def append1(self, *transition):
        # (state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition)))
        print('Append buffer size', len(self.buffer))

    def sample1(self, batch_size, device):
        '''sample a batch of transition tensors'''
        print('Sample buffer size', len(self.buffer))
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))

class Net(nn.Module):
    def __init__(self, state_dim=8, action_dim=4):
        """Model Blueprint
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, action_dim)
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_vals = self.out(x)
        return q_vals

class DDQN:
    """Define DDQN Agent"""
    def __init__(self, args):
        self.seed = torch.manual_seed(args.seed)
        self._behavior_net = Net().to(args.device)
        self._target_net = Net().to(args.device)
        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        self._optimizer = optim.Adam(self._behavior_net.parameters(), lr=args.lr)
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq

    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network
        Params
        ======
            state (array_like): current state
            episilon (float): epsilon, for epsilon-greedy action selection
            action_space (array): available actions
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self._behavior_net.eval()
        with torch.no_grad():
            action_values = self._behavior_net(state)
        self._behavior_net.train()

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            #return np.argmax(action_values.cpu().data.numpy())
            return torch.argmax(action_values).cpu().data.numpy()
        else:
            return action_space.sample()

    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, [action], [reward / 10], next_state,
                            [int(done)])

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(self.batch_size, self.device)
        # compute Q_target from behavior network inputing next state
        with torch.no_grad():
            # mini-batch
            action_values = self._behavior_net(next_state).detach()
            #print('action_values', action_values)
            argmax_action_values = torch.argmax(action_values)
            #print('argmax action_values', argmax_action_values)
            Q_index = argmax_action_values.cpu().data.numpy()
            #Q_index = torch.argmax(self._behavior_net(next_state)).cpu().data.numpy()
            Q_target_av = self._target_net(next_state).detach().view(-1)[Q_index]

            #print('Q_target_av', Q_target_av)
            Q_target = reward + gamma*(Q_target_av)*(1-done) # broadcasting works here.
            #print('Q_target', Q_target)
        Q_value = self._behavior_net(state).gather(1, action)
        #print('Q_value', Q_value)
        loss = F.mse_loss(Q_value, Q_target)
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)
        self._optimizer.step()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def _update_target_network(self):
        '''update target network by copying from behavior network'''
        self.soft_update(self._behavior_net, self._target_net, 1e-3)

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'behavior_net': self._behavior_net.state_dict(),
                    'target_net': self._target_net.state_dict(),
                    'optimizer': self._optimizer.state_dict(),
                }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])


def train(args, env_name, agent, writer):
    print('Start Training')
    env = gym.make(env_name)
    action_space = env.action_space
    total_steps, epsilon, ewma_reward = 0, 1., 0.
    writer.add_graph(agent._behavior_net, torch.FloatTensor([0., 0., 0., 0., 0., 0., 0., 0.]).to(args.device), True)
    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        for t in itertools.count(start=1):
            # select action
            if total_steps < args.warmup:
                action = action_space.sample()
            else:
                action = agent.select_action(state, epsilon, action_space)
                #epsilon = max(epsilon * args.eps_decay, args.eps_min)
            # execute action
            next_state, reward, done, _ = env.step(action)
            # store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                agent.update(total_steps)

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Epsilon', epsilon, total_steps)
                writer.add_scalar('Train/Episode Reward', total_reward,
                                  total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward,
                                  total_steps)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                    .format(total_steps, episode, t, total_reward, ewma_reward,
                            epsilon))
                break
        if total_steps >= args.warmup:
            epsilon = max(epsilon * args.eps_decay, args.eps_min)
        if episode % 500 == 0:
            agent.save('checkpoints/ddqn-{}.pth'.format(episode))
    env.close()
    return ewma_reward


def test(args, env_name, agent, writer):
    print('Start Testing')
    env = gym.make(env_name)
    action_space = env.action_space
    epsilon = args.test_epsilon
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()
        for j in range(env._max_episode_steps):
            action = agent.select_action(state, epsilon, action_space)
            if args.render:
                env.render()
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                break
        rewards.append(total_reward)
    print('Average Reward', np.mean(rewards))
    #writer.add_hparams(args.__dict__,{'Test/Average Reward': np.mean(rewards)})
    env.close()


def main():
    _current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # train
    parser.add_argument('--warmup', default=10000, type=int,
                        help='number of warmup steps')
    parser.add_argument('--episode', default=1200, type=int,
                        help='upper limit of training episodes')
    parser.add_argument('--capacity', default=10000, type=int,
                        help='capacity of replay buffer')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='mini batch size extract from replay buffer')
    parser.add_argument('--lr', default=.0005, type=float,
                        help='learning rate')
    parser.add_argument('--eps_decay', default=.995, type=float,
                        help='epsilon decay rate')
    parser.add_argument('--eps_min', default=.01, type=float,
                        help='lower bound of epsilon')
    parser.add_argument('--gamma', default=.99, type=float,
                        help='gamma for update Q value')
    parser.add_argument('--freq', default=4, type=int,
                        help='interval to update behavior network')
    parser.add_argument('--target_freq', default=1000, type=int,
                        help='interval to update target network')
    # test
    parser.add_argument('--test_only', action='store_true',
                        help='conduct test only runs')
    parser.add_argument('--render', default=False, action='store_true',
                        help='render display')
    parser.add_argument('--test_epsilon', default=.001, type=float,
                        help='test epsilon')
    # utilities
    parser.add_argument('-d', '--device', default='cuda',
                        help='device used for training / testing')
    parser.add_argument('-m', '--model', default='models/ddqn-{}.pth'.format(_current_datetime),
                        help='path to pretrained model / model save path')
    parser.add_argument('--logdir', default='log/ddqn/{}'.format(_current_datetime),
                        help='path to tensorboard log')
    parser.add_argument('--seed', default=2021111, type=int,
                        help='random seed')
    args = parser.parse_args()

    ## main ##
    env_name = 'LunarLander-v2'
    agent = DDQN(args)
    writer = SummaryWriter(args.logdir)
    if not args.test_only:
        ewma_reward = train(args, env_name, agent, writer)
        #writer.add_hparams(args.__dict__,{'Train/Final Ewma Reward': ewma_reward})
        agent.save(args.model)
    agent.load(args.model)
    test(args, env_name, agent, writer)


if __name__ == '__main__':
    main()
