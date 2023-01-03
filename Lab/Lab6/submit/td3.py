'''DLP TD3 Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from os.path import exists
import copy

class GaussianNoise:
    def __init__(self, dim, mu=None, std=None):
        self.mu = mu if mu else np.zeros(dim)
        self.std = std if std else np.ones(dim) * .1

    def sample(self):
        # print('rnd: ', np.random.normal(self.mu, self.std))
        return np.random.normal(self.mu, self.std)


class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        ## TODO ##
        transitions = random.sample(self.buffer, batch_size)  # sample: (state(8), action(2), reward(1), next_state(8), done(1)) * batch_size
        # unzip transitions to state_vector, action_vector, reward_vector, next_state_vector, done_vector
        return (torch.tensor(out, dtype=torch.float, device=device) for out in zip(*transitions))  # transfer from double to torch.float
        # raise NotImplementedError


class ActorNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
        super().__init__()
        ## TODO ##
        h1, h2 = hidden_dim
        self.actor = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, action_dim),
            nn.Tanh(),
        )

        # raise NotImplementedError

    def forward(self, x):
        ## TODO ##
        return self.actor(x)
        # raise NotImplementedError


class CriticNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=256):
        super(CriticNet, self).__init__()
        self.relu = nn.ReLU()
        # critic1
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, 1)

        # critic2
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x1 = self.q1_fc1(x)
        x1 = self.relu(x1)
        x1 = self.q1_fc2(x1)
        x1 = self.relu(x1)
        x1 = self.q1_fc3(x1)

        x2 = self.q2_fc1(x)
        x2 = self.relu(x2)
        x2 = self.q2_fc2(x2)
        x2 = self.relu(x2)
        x2 = self.q2_fc2(x2)
        return x1, x2  # output critic1 and 2

    def single_q(self, state, action):
        x = torch.cat([state, action], 1)

        x = self.relu(self.q1_fc1(x))
        x = self.relu(self.q1_fc2(x))
        x = self.q1_fc3(x)
        return x  # output single critic


class TD3(object):
    def __init__(self, args):
        ## config ##
        self.max_action = args.action_max
        self.gamma = args.gamma
        self.tau = args.tau
        self.noise = args.var
        self.cp = args.cp
        self.update_step = args.update_step
        self.device = args.device
        self.batch_size = args.batch_size
        # behavior network
        self._actor_net = ActorNet().to(self.device)
        self._critic_net = CriticNet().to(self.device)
        # target network
        self._target_actor_net = copy.deepcopy(self._actor_net)
        self._target_critic_net = copy.deepcopy(self._critic_net)
        ## TODO ##
        self._actor_opt = torch.optim.Adam(self._actor_net.parameters(), lr=args.lra)
        self._critic_opt = torch.optim.Adam(self._critic_net.parameters(), lr=args.lrc)
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

    def select_action(self, state):
        '''based on the behavior (actor) network and exploration noise'''
        ## TODO ##
        state = torch.Tensor(state).to(self.device)
        return self._actor_net(state).cpu().detach().numpy()

    def _update_network(self, current_step, gamma):
        actor_net, critic_net, target_actor_net, target_critic_net = self._actor_net, self._critic_net, self._target_actor_net, self._target_critic_net
        actor_opt, critic_opt = self._actor_opt, self._critic_opt

        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(self.batch_size, self.device)
        ## update critic ##
        # critic loss
        ## TODO ##
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.noise).clamp(-self.cp, self.cp)  # 3nd: select action and clip noise
            next_action = (target_actor_net(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute q_target value
            q1_next, q2_next = target_critic_net(next_state, next_action)
            q_next = torch.min(q1_next, q2_next)  # 2nd: min(double critic)
            q_target = reward + gamma * q_next * (1 - done)

        q1_value, q2_value = critic_net(state, action)  # get q_value from both critic
        critic_loss = F.mse_loss(q1_value, q_target) + F.mse_loss(q2_value, q_target)

        # optimize critic
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        # 2nd Delayed updates
        if current_step % self.update_step == 0:
            ## update actor ##
            # actor loss
            ## TODO ##
            actor_loss = -critic_net.single_q(state, actor_net(state)).mean()

            # optimize actor
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            '''update target network by _soft_ copying from behavior network'''
            ## TODO ##
            for param, target_param in zip(critic_net.parameters(), target_critic_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(actor_net.parameters(), target_actor_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, action, [reward / 100], next_state, [int(done)])

    def update(self, current_step):
        # update the behavior networks
        self._update_network(current_step, self.gamma)

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
        model = torch.load(model_path)
        self._actor_net.load_state_dict(model['actor'])
        self._critic_net.load_state_dict(model['critic'])
        if checkpoint:
            self._target_actor_net.load_state_dict(model['target_actor'])
            self._target_critic_net.load_state_dict(model['target_critic'])
            self._actor_opt.load_state_dict(model['actor_opt'])
            self._critic_opt.load_state_dict(model['critic_opt'])


def train(args, env, agent, writer):
    print('Start Training')
    total_steps = 0
    ewma_reward = 0
    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()  # state/observation: total 8 state: hori , vert coordinate, hori, vert speed, angle, angle speed, first leg, second leg contact
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
                agent.update(total_steps)

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward,
                                  total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward,
                                  total_steps)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}'
                    .format(total_steps, episode, t, total_reward,
                            ewma_reward))
                break
    env.close()


def test(args, env, agent, writer):
    print('Start Testing')
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()
        ## TODO ##
        for t in itertools.count(start=1):
            # select action
            action = agent.select_action(state)
            # execute action
            next_state, reward, done, _ = env.step(action)  # done - end game
            # store transition
            agent.append(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                print('Episode: {}\tLength: {:3d}\tTotal reward: {:.2f}'.format(n_episode, t, total_reward))
                rewards.append(total_reward)
                break
    ave_rewards = np.mean(rewards)
    print('Average Reward: {}, Average/30: {}'.format(ave_rewards, ave_rewards / 30))
    env.close()


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='td3.pth')
    parser.add_argument('--logdir', default='log/td3')
    parser.add_argument('--model_dir', default='model')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=1200, type=int)
    parser.add_argument('--update_step', default=2, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--capacity', default=500000, type=int)
    parser.add_argument('--lra', default=1e-3, type=float)  #
    parser.add_argument('--lrc', default=2e-4, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--tau', default=.005, type=float)
    parser.add_argument('--var', default=.1, type=float)  # .1
    parser.add_argument('--cp', default=.4, type=float)  # .4
    parser.add_argument('--action_max', default=0, type=float)
    # test
    parser.add_argument('--test_only', default=False, action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20200519, type=int)
    args = parser.parse_args()

    ## main ##
    print(args)
    env = gym.make('LunarLanderContinuous-v2')
    args.action_max = env.action_space.high[0]
    agent = TD3(args)
    writer = SummaryWriter(args.logdir)
    path = '%s/%s' % (args.model_dir, args.model)
    if exists(path):
        agent.load(path)
        print('loaded model')
    if not args.test_only:
        train(args, env, agent, writer)
        agent.save(args.model)
    agent.load(args.model)
    test(args, env, agent, writer)


if __name__ == '__main__':
    main()
