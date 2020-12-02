'''DLP DDPG Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque, namedtuple
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from datetime import datetime
import os


class GaussianNoise:
    def __init__(self, dim, mu=None, std=None):
        self.mu = mu if mu else np.zeros(dim)
        self.std = std if std else np.ones(dim) * .1

    def sample(self):
        return np.random.normal(self.mu, self.std)


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
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)


class ActorNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
        super().__init__()
        h1, h2 = hidden_dim
        self.actor_head = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.actor_head(x)
        x = self.actor(x)
        return x


class CriticNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
        super().__init__()
        h1, h2 = hidden_dim
        self.critic_head = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU()
        )
        # https://arxiv.org/pdf/1509.02971.pdf
        # Actions were not included until the 2nd hidden layer of Q.
        self.critic = nn.Sequential(
            nn.Linear(h1 + action_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, 1)
        )
        # https://arxiv.org/pdf/1804.00361.pdf (adding layer norm)

    def forward(self, state, action):
        x = self.critic_head(state)
        x = self.critic(torch.cat([x, action], dim=1))
        return x


class DDPG:
    def __init__(self, state_dim, action_dim, args, writer):
        # behavior network
        self._actor_net = ActorNet(state_dim=state_dim, action_dim=action_dim).to(args.device)
        self._critic_net = CriticNet(state_dim=state_dim, action_dim=action_dim).to(args.device)
        # target network
        self._target_actor_net = ActorNet(state_dim=state_dim, action_dim=action_dim).to(args.device)
        self._target_critic_net = CriticNet(state_dim=state_dim, action_dim=action_dim).to(args.device)
        # initialize target network
        self._target_actor_net.load_state_dict(self._actor_net.state_dict())
        self._target_critic_net.load_state_dict(self._critic_net.state_dict())
        self._actor_opt = optim.Adam(self._actor_net.parameters(), lr=args.lra)
        self._critic_opt = optim.Adam(self._critic_net.parameters(), lr=args.lrc, weight_decay=1e-2)
        # weight decay of 10−2 from https://arxiv.org/pdf/1509.02971.pdf
        # action noise
        self._action_noise = GaussianNoise(dim=action_dim)
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.gamma = args.gamma
        self.writer = writer
        self.num_actor_update_iteration = 0
        self.num_critic_update_iteration = 0

    def select_action(self, state, noise=True):
        '''based on the behavior (actor) network and exploration noise'''
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self._actor_net(state).cpu().data.numpy().flatten()
        if noise:
            action = (action + self._action_noise.sample())
        action = np.clip(action, -1, 1)
        return action

    def append(self, state, action, reward, next_state, done):
        # https://www.quora.com/In-DQN-should-reward-be-normalized-and-standardized
        # Standardizing rewards usually helps as they keep the gradients that are
        # being back-propagated and the Q-values of the actions from saturating or blowing up.
        # https://stackoverflow.com/questions/49801638/normalizing-rewards-to-generate-returns-in-reinforcement-learning
        #self._memory.append(state, action, [reward / 100], nedxt_state,
        #[int(done)])
        self._memory.append(state, action, reward, next_state, int(done))

    def update(self):
        # update the behavior networks
        self._update_behavior_network(self.gamma)
        # update the target networks
        self._update_target_network(self._target_actor_net, self._actor_net,
                                    self.tau)
        self._update_target_network(self._target_critic_net, self._critic_net,
                                    self.tau)
        self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1

    def _update_behavior_network(self, gamma):
        actor_net, critic_net, target_actor_net, target_critic_net = self._actor_net, self._critic_net, self._target_actor_net, self._target_critic_net
        actor_opt, critic_opt = self._actor_opt, self._critic_opt

        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)

        ## update critic ##
        # critic loss
        # Compute the target Q value
        with torch.no_grad():
            target_Q = target_critic_net(next_state, target_actor_net(next_state))
            target_Q = reward + self.gamma * (1 - done) * target_Q
            #target_Q = reward + self.gamma * target_Q
        # Get current Q estimate
        current_Q = critic_net(state, action)
        # critic_loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.writer.add_scalar('Train/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
        # optimize critic
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        ## update actor ##
        # actor loss
        actor_loss = (- critic_net(state, actor_net(state)) ).mean()
        self.writer.add_scalar('Train/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
        # optimize actor
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

    @staticmethod
    def _update_target_network(target_net, behavior_net, tau):
        '''update target network by _soft_ copying from behavior network'''
        for target, behavior in zip(target_net.parameters(), behavior_net.parameters()):
            target.data.copy_(tau * behavior.data + (1 - tau) * target.data)

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'actor': self._actor_net.state_dict(),
                    'critic': self._critic_net.state_dict(),
                    'target_actor': self._target_actor_net.state_dict(),
                    'target_critic': self._target_critic_net.state_dict(),
                    'actor_opt': self._actor_opt.state_dict(),
                    'critic_opt': self._critic_opt.state_dict(),
                }, model_path)
        else:
            torch.save(
                {
                    'actor': self._actor_net.state_dict(),
                    'critic': self._critic_net.state_dict(),
                }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path, torch.device(self.device))
        self._actor_net.load_state_dict(model['actor'])
        self._critic_net.load_state_dict(model['critic'])
        if checkpoint:
            self._target_actor_net.load_state_dict(model['target_actor'])
            self._target_critic_net.load_state_dict(model['target_critic'])
            self._actor_opt.load_state_dict(model['actor_opt'])
            self._critic_opt.load_state_dict(model['critic_opt'])


def train(args, env_name, agent, writer):
    print('Start Training')
    env = gym.make(env_name)
    total_steps, ewma_reward = 0, 0.
    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        for t in itertools.count(start=1):
            # select action
            if total_steps < args.warmup:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            # execute action
            next_state, reward, done, _ = env.step(action)
            # store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                agent.update()

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward, episode)
                writer.add_scalar('Train/Ewma Reward', ewma_reward, episode)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}'
                    .format(total_steps, episode, t, total_reward,
                            ewma_reward))
                break
        if episode % 100 == 0:
            agent.save('checkpoints/ddpg-{}.pth'.format(episode), checkpoint=True)
    env.close()
    return ewma_reward


def test(args, env_name, agent, writer):
    print('Start Testing')
    env = gym.make(env_name)
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()
        for j in range(env._max_episode_steps):
            action = agent.select_action(state, noise=False)
            if args.render:
                env.render()
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                break
        rewards.append(total_reward)
    print('Average Reward', np.mean(rewards))
    env.close()

class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)

def main():
    _current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # train
    parser.add_argument('--warmup', default=10000, type=int,
                        help='number of warmup steps')
    parser.add_argument('--episode', default=10000, type=int,
                        help='upper limit of training episodes')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='mini batch size extract from replay buffer')
    parser.add_argument('--capacity', default=1000000, type=int,
                        help='capacity of replay buffer')
    parser.add_argument('--lra', default=1e-4, type=float,
                        help='learning rate actor')
    parser.add_argument('--lrc', default=1e-3, type=float,
                        help='learning rate critic')
    parser.add_argument('--gamma', default=.99, type=float,
                        help='gamma for update Q value')
    parser.add_argument('--tau', default=.001, type=float,
                        help='soft update ratio')
    parser.add_argument('--load_checkpoint', default="", type=str,
                        help='specify checkpoint path')
    # test
    parser.add_argument('--test_only', action='store_true',
                        help='conduct test only runs')
    parser.add_argument('--render', default=False, action='store_true',
                        help='render display')
    # global config
    parser.add_argument('-d', '--device', default='cuda',
                        help='device used for training / testing')
    parser.add_argument('-m', '--model', default='models/ddpg-{}.pth'.format(_current_datetime),
                        help='path to pretrained model / model save path')
    parser.add_argument('--logdir', default='log/ddpg/{}'.format(_current_datetime),
                        help='path to tensorboard log')
    parser.add_argument('--seed', default=2021111, type=int,
                        help='random seed')
    parser.add_argument('--env', default='LunarLanderContinuous-v2', type=str,
                        help='environment name')
    args = parser.parse_args()

    ## main ##
    env_name = args.env
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print('Action space', env.action_space)
    print('Observation space', env.observation_space)
    writer = SummaryWriter(args.logdir)
    agent = DDPG(state_dim, action_dim, args, writer)
    if not args.test_only:
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        if args.load_checkpoint:
            print('load check point', args.load_checkpoint)
            agent.load(args.load_checkpoint, checkpoint=True)
        ewma_reward = train(args, env_name, agent, writer)
        writer.add_hparams(args.__dict__,{'hparam/final_ewma_reward': ewma_reward})
        agent.save(args.model)
    agent.load(args.model)
    test(args, env_name, agent, writer)


if __name__ == '__main__':
    main()
