import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from modules.ou_noise import OUNoise
from modules.gaussian_noise import GaussianNoise
import modules.evaluate as E
from modules.layer import HGRU
from modules.replay_buffer import ReplayBuffer, ReplayBufferGPU
from modules.actor_critic import PolicyNetwork, ValueNetwork, Discriminator, CNNPolicy, VariationalPolicy
from modules.loss import VAELoss, LossFunction
from tqdm import tqdm
import matplotlib.pyplot as plt
from modules.loss import MrrPolicyLoss

# inspired in  https://github.com/higgsfield/RL-Adventure-2/blob/master/5.ddpg.ipynb

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
MODELS_PATH = '/media/gustavo/Storage/gustavo/repository/icmc-mestrado-code/models/'
MODEL_FILE = 'last_fm_TOP1Max_Adagrad_0.1_epoch1'


class NormalizedActions(gym.ActionWrapper):

    def _action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)

        return action

    def _reverse_action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)

        return action


def normalize_vector(vector, low_bound=-1.0, upper_bound=1.0):
    action = 2.0 * (vector - low_bound) / (upper_bound - low_bound) - 1.0
    action = torch.clamp_(action, low_bound, upper_bound)
    return action.detach()


def plot(frame_idx, rewards):
    plt.figure(figsize=(5, 5))
    ax1 = plt.subplot(311)
    plt.plot(frame_idx, rewards[:, 2])
    # ax2 = plt.subplot(312, )
    plt.plot(frame_idx, rewards[:, 1])
    # ax2 = plt.subplot(313, )
    plt.plot(frame_idx, rewards[:, 0])
    plt.show()


class DDPG(object):

    def __init__(self, state_dim, action_dim, hidden_dim, gamma=0.99, soft_tau=1e-2,
                 noise_mode='ounoise', strategy='replace', lr_policy=1e-6, lr_value=1e-6,
                 policy_updates=1, reset_buffer_limit=-1, model_name='ddpg', device='cuda'):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        self.replay_buffer_size = 3500000
        self.strategy = strategy
        # noise generator
        self.noise_mode = noise_mode
        # soft-hard update params
        self.soft_tau = soft_tau
        self.gamma = gamma
        # Learinng rate
        self.value_lr = lr_value
        self.policy_lr = lr_policy
        self.init_networks()
        self.init_replay_buffer()
        self.init_criterion()
        self.init_noise_generator()
        self.policy_updates = policy_updates
        self.reset_buffer_limit = reset_buffer_limit
        r_buffer_l = 'r' + str(reset_buffer_limit) if reset_buffer_limit > 0 else ''
        p_updates = 'u' + str(policy_updates) if policy_updates > 1 else ''
        self.info_name = model_name + 's' + str(state_dim) + 'a' + str(action_dim) + 'h' + str(
            hidden_dim) + 'st' + strategy + p_updates + r_buffer_l

    def init_noise_generator(self, gaussian_std=1e-12):
        if self.noise_mode == 'ounoise':
            l = 0 if self.strategy == 'gate-init' else -1
            self.noise_generator = OUNoise(action_space=self.action_dim, low=l)
        elif self.noise_mode == 'gaussian':
            self.noise_generator = GaussianNoise(self.action_dim, gaussian_std)
        else:
            raise NotImplementedError('Noise generator mode not implemented')

    def init_criterion(self):
        # Optmizers
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)
        # Criterion
        self.value_criterion = nn.MSELoss()
        self.policy_criterion = VAELoss()

    def init_replay_buffer(self):
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

    def init_networks(self):

        self.value_net = ValueNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim,
                                        strategy=self.strategy).to(self.device)

        self.target_value_net = ValueNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim,
                                               strategy=self.strategy).to(self.device)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self):

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

    def print_parameters(self):
        policy_params = ['policy:']
        value_params = ['value:']
        for name, param in self.target_policy_net.named_parameters():
            if name.startswith('linear'):
                policy_params.append('{:5.5}:{:.4f}'.format(name, param.norm().item()))
        print(' '.join(policy_params))
        for name, param in self.target_value_net.named_parameters():
            if name.startswith('linear'):
                value_params.append('{:5.5}:{:.4f}'.format(name, param.norm().item()))
        print(' '.join(value_params))

    def load_policy(self, url):
        self.policy_net.load_state_dict(torch.load(url + '_target_policy_net'))
        self.policy_net.eval()
        return self.policy_net

    def ddpg_update(self, batch_size, min_value=0, max_value=1.0):

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)

        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * (self.gamma * target_value)
        expected_value = torch.clamp(expected_value, min_value, max_value)
        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())
        loss = value_loss
        if torch.isnan(loss):
            print(action)
            print(reward)
            print(done)
            print(state)
            print(policy_loss)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.soft_update()
        # self.print_parameters()

        return loss

    def ddpg_update_all(self, batch_size, min_value=0, max_value=1.0):
        ploss, vloss=0.0,0.0
        iters=0
        for state, action, reward, next_state, done  in  self.replay_buffer:
            iters=+1
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)

            action = torch.FloatTensor(action).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device)
            done = torch.FloatTensor(done).to(self.device)

            policy_loss = self.value_net(state, self.policy_net(state))
            policy_loss = -policy_loss.mean()
            ploss= ploss+policy_loss
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            next_action = self.target_policy_net(next_state)
            target_value = self.target_value_net(next_state, next_action.detach())
            expected_value = reward + (1.0 - done) * (self.gamma * target_value)
            expected_value = torch.clamp(expected_value, min_value, max_value)
            value = self.value_net(state, action)
            value_loss = self.value_criterion(value, expected_value.detach())
            vloss= vloss + value_loss
            if torch.isnan(vloss):
                print(action)
                print(reward)
                print(done)
                print(state)
                print(policy_loss)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            self.soft_update()
            # self.print_parameters()
        return  ploss/iters,vloss/iters

    def rollout(self, h_s, model, dataloader, batch_id, max_steps=3, k=5):
        # h_s shape : [1 x batch_size x hidden_dim]
        dummy = torch.zeros(h_s.shape[1]).to(self.device).view(-1, 1).float()
        rewards_recall, rewards_mrr = [dummy], [dummy]
        count_steps = torch.zeros(h_s.shape[1], 1).to(self.device)

        batch_id_start, batch_id_end = batch_id, batch_id + max_steps
        data = dataloader.get_batches(batch_id_start, batch_id_end)
        count_vec = torch.ones(count_steps.shape).to(self.device)
        for input, target, Sstart, Ustart, in_eval in data:
            sstart = torch.Tensor(Sstart).to(self.device).view(-1, 1)
            input = input.to(self.device)
            target = target.to(self.device)
            mask_ended_sesions = sstart.nonzero()
            count_vec[mask_ended_sesions[:, 0]] = 0.0
            count_steps += count_vec

            out, h_s = model(input, h_s)

            step_reward_rec, step_reward_mrr = E.evaluate(out, target, k=k, batch_wise=True)
            rewards_mrr.append(step_reward_mrr * count_vec)
            rewards_recall.append(step_reward_rec.float() * count_vec)
            max_steps -= 1
            if max_steps == 0:
                break

        rollout_rewards_mrr = torch.cat(rewards_mrr, dim=1).sum(dim=1).view(-1, 1)
        rollout_rewards_rec = torch.cat(rewards_recall, dim=1).sum(dim=1).view(-1, 1)
        count_steps[(rollout_rewards_mrr == 0.0).nonzero()] = 1.0

        rollout_rewards_mrr /= count_steps * 1.0
        rollout_rewards_rec /= count_steps * 1.0

        return rollout_rewards_mrr, rollout_rewards_rec

    def rollout_on_policy(self, h_s, h_u, h_s_init, model, policy, dataloader, batch_id,
                          max_steps=3, k=5, min_steps=3, min_sessions=3):
        # h_s shape : [1 x batch_size x hidden_dim]
        dummy = torch.zeros(h_s.shape[1]).to(self.device).view(-1, 1).float()
        rewards_recall, rewards_mrr = [dummy], [dummy]
        count_steps = torch.zeros(h_s.shape[1], 1).to(self.device)

        batch_id_start, batch_id_end = batch_id, batch_id + max_steps
        data = dataloader.get_batches(batch_id_start, batch_id_end)
        count_vec = torch.ones(count_steps.shape).to(self.device)
        for input, target, Ustart, Sstart, in_eval in data:
            sstart = torch.Tensor(Sstart).to(self.device).view(-1, 1)
            input = input.to(self.device)
            target = target.to(self.device)
            mask_ended_sesions = sstart.nonzero()
            count_vec[mask_ended_sesions[:, 0]] = 0.0
            count_steps += count_vec
            # Bootstrapping the policy and the environment  max_steps times
            steps, session_count, mask_for_interactions = generate_interaction_mask(steps, session_count,
                                                                                    Sstart, Ustart,
                                                                                    min_steps, min_sessions,
                                                                                    in_eval['in_rl'])
            state = model.get_state(input, Sstart, Ustart, h_s, h_u, h_s_init, steps)
            out, h_s, h_u = model.forward_discriminator(input, Sstart, Ustart, h_s, h_u, h_s_init,
                                                        mask_for_interactions, steps)

            policy.forward(state)
            h_s_init, h_u = model.user_gru(h_s, h_u)

            step_reward_rec, step_reward_mrr = E.evaluate(out, target, k=k, batch_wise=True)
            rewards_mrr.append(step_reward_mrr * count_vec)
            rewards_recall.append(step_reward_rec.float() * count_vec)
            max_steps -= 1
            if max_steps == 0:
                break

        rollout_rewards_mrr = torch.cat(rewards_mrr, dim=1).sum(dim=1).view(-1, 1)
        rollout_rewards_rec = torch.cat(rewards_recall, dim=1).sum(dim=1).view(-1, 1)
        count_steps[(rollout_rewards_mrr == 0.0).nonzero()] = 1.0

        # rollout_rewards_mrr /= count_steps * 1.0
        # rollout_rewards_rec /= count_steps * 1.0

        return rollout_rewards_mrr, rollout_rewards_rec

    def save_netwokrs(self, model_dir, name):
        torch.save(self.value_net.state_dict(), model_dir + name + '_value_net')
        torch.save(self.policy_net.state_dict(), model_dir + name + '_policy_net')
        torch.save(self.target_value_net.state_dict(), model_dir + name + '_target_value_net')
        torch.save(self.target_policy_net.state_dict(), model_dir + name + '_target_policy_net')


def generate_interaction_mask(steps, session_count, sstart, ustart, min_steps, min_sessions,
                              valid_rl_rows):
    # print(valid_rl_rows)
    valid_rl_rows = valid_rl_rows.view(-1, 1)
    steps *= (1.0 - sstart)
    steps += 1.0
    session_count *= (1.0 - ustart)
    session_count += sstart
    mask_for_interactions = ((steps % min_steps == 0) & (steps >= min_steps)).float()
    mask_for_interactions = valid_rl_rows * mask_for_interactions * (session_count > min_sessions).float()
    return steps, session_count, mask_for_interactions


def run_ddpg(env, model, loader, model_dir='/models', cutoff=10, name='ddpg', batch_size=32, max_frames=1000000,
             rollout=3, min_mean_reward=.2, min_steps=3, min_sessions=3):
    error = False
    device = model.device
    frame_idx = 0
    rewards = []
    temp_loader = loader
    loader.generate_batches()

    while len(model.replay_buffer) < max_frames:
        loader = temp_loader
        losses_ddpg = []
        # This should load the same original model  restarting  users' sessions
        state = env.get_reset_state(batch_size, model.state_dim)
        state_env = env.get_reset_state(batch_size, model.state_dim)
        # This should reset the noise generator for exploration states
        model.noise_generator.reset()
        # This will store the total reward of the frame(episode)
        episode_reward, episode_recall, episode_intra = torch.zeros(batch_size, 1).to(model.device), \
                                                        torch.zeros(batch_size, 1).to(model.device), \
                                                        torch.zeros(batch_size, 1).to(model.device)
        count = 0.0
        loss = 0.0
        steps = torch.ones(batch_size, 1).to(device)
        session_count = torch.ones(batch_size, 1).to(device)
        Hs, Hu = env.session_gru.init_hidden(), env.user_gru.init_hidden()
        Hs_step, Hu_step = env.session_gru.init_hidden(), env.user_gru.init_hidden()
        next_state_iterator = iter(loader)
        next_x, next_y, next_Ustart, next_Sstart, next_in_eval = next(next_state_iterator)
        session_steps = torch.zeros(batch_size, 1).to(model.device)
        session_reward = torch.zeros(batch_size, 1).to(model.device)
        session_reward_env = torch.zeros(batch_size, 1).to(model.device)
        skip = torch.ones(batch_size, 1).to(model.device)
        batch_id = 0
        H_s_init = torch.zeros(batch_size).to(model.device)
        for input, target, Sstart, Ustart, in_eval in tqdm(loader.batches):

            input = input.to(model.device)
            target = target.to(model.device)
            # Forward the concatenation of inputs (h_u, Embed(x) or ((h_u, h_s, Embed(x))
            # This 'action' is the new improved  hidden state which  is generated by the policy
            action = model.policy_net.get_action(state).detach()
            if action.isnan():
                raise RuntimeError(" nan action")
            # Add noise to  action
            action = model.noise_generator.get_action_batch(action.detach(), steps).detach()
            # Calculate new state and reward values

            Sstart = torch.Tensor(Sstart).to(model.device).view(-1, 1)
            Ustart = torch.Tensor(Ustart).to(model.device).view(-1, 1)

            # We count the number of the steps to create an episode each k steps for long sessions
            # steps accumulates the number of intra-sessions interactions to then select the rows
            # to be stored in replay buffer

            # steps *= (1.0 - Sstart)
            # steps += 1.0
            # session_count *= (1.0 - Ustart)
            # session_count += 1.0
            # mask_for_interactions = ((steps % min_steps >= 0)& (steps>min_steps)) .float()
            # mask_for_interactions = mask_for_interactions * (session_count > min_sessions).float()

            # valid_rl_rows = torch.Tensor(in_eval['in_rl']).to(model.device).view(-1,1).float()
            valid_rl_rows = torch.ones(batch_size).to(model.device)
            steps, session_count, mask_for_interactions = generate_interaction_mask(steps,
                                                                                    session_count,
                                                                                    Sstart, Ustart,
                                                                                    min_steps,
                                                                                    min_sessions,

                                                                                    valid_rl_rows
                                                                                    )
            H_s_init = (Hs[-1] * Sstart) * (1 - Ustart)
            logit_step, Hu_step, Hs_step = env.step(input, Sstart, Ustart, Hs.detach(), Hu.detach(), action,
                                                    mask_for_interactions, model.strategy)
            logit, Hu, Hs = env(input, Sstart, Ustart, Hs.detach(), Hu.detach())

            logit_sampled = logit[:, target.view(-1)]

            logit_sampled_step = logit_step[:, target.view(-1)]

            try:
                next_x, next_y, next_Ustart, next_Sstart, next_in_eval = next(next_state_iterator)
            except StopIteration:
                print('got inside iterator')
                break

            # Possible reward functions
            # recall, mrr = E.evaluate(logit_sampled_step, target, cutoff, batch_wise=True)
            recall_env, mrr_env = model.rollout(Hs, env.session_gru, loader, batch_id, max_steps=rollout, k=cutoff)

            mrr, recall = model.rollout(Hs_step, env.session_gru, loader, batch_id, max_steps=rollout, k=cutoff)

            batch_id += 1

            reward = mrr

            session_reward += mrr * (1.0 - Sstart) + mrr * (Sstart)
            session_reward_env += mrr_env
            mean_session_reward_env = session_reward_env / steps

            mean_session_reward = session_reward / steps
            count += mask_for_interactions.sum()

            episode_reward += reward.sum()
            episode_recall += recall
            episode_intra += reward * mask_for_interactions
            frame_idx += mask_for_interactions.sum()

            done = (mean_session_reward < mean_session_reward_env)
            session_steps = session_steps * (1.0 - Sstart) + (1.0 - Sstart)

            next_Sstart = torch.Tensor(next_Sstart).to(device).view(-1, 1)
            next_Ustart = torch.Tensor(next_Ustart).to(device).view(-1, 1)
            next_x = next_x.to(device)

            next_steps = steps * (1.0 - next_Sstart)
            next_steps += 1.0
            # next_Sstart *= 0.0# trick to intra-session  user level propagation
            next_state = env.get_state(next_x, next_Sstart, next_Ustart, Hs_step.detach(), Hu_step.detach(),
                                       H_s_init.detach(),
                                       False, model.strategy, next_steps).to(device)

            # next_state_env = env.get_state(next_x, next_Sstart, next_Ustart, Hs.detach(), Hu.detach()
            #                               , return_x_state=False).to(device)

            # Push transitions to replay buffer

            with torch.no_grad():
                # print([x.mean() for x in [state, action, mean_session_reward, next_state, mask_for_interactions]])
                model.replay_buffer.push_batch(state.cpu(), action.cpu(), session_reward.cpu(), next_state.cpu(),
                                               done.cpu(), mask_for_interactions.flatten().cpu())
                # model.replay_buffer.push_batch(state, action, session_reward, next_state,
                #                               done, mask_for_interactions.flatten())
                # model.replay_buffer.push_batch(state_env.cpu(), torch.zeros(batch_size, 1), steps_reward_env.cpu(), next_state_env.cpu(), done.cpu(),
                #  1.0-mask_for_interactions)

            state = next_state
            # if int(frame_idx) > len(model.replay_buffer):
            #     print('RB_size: {} ,frame_idx: {}'.format(len(model.replay_buffer),frame_idx))
            #     print(mask_for_interactions.flatten())
            #     print(next_state.mean(1).flatten())
            if batch_id % 1e3 == 0:
                print('RB_size: {} ,frame_idx: {}'.format(len(model.replay_buffer), frame_idx, str(H_s_init.size())))
            if len(model.replay_buffer) > 2 * batch_size:
                for u in range(model.policy_updates):
                    loss = model.ddpg_update(batch_size)

                    losses_ddpg.append(loss)

                    if torch.isnan(loss):
                        print('RB_size: {} ,frame_idx: {}'.format(len(model.replay_buffer), frame_idx))
                        raise RuntimeError('value_loss nan')
                # mean_loss = loss.mean().item()
                # losses_ddpg.append(mean_loss)
                if int(frame_idx) % 1e4 == 0:
                    model.print_parameters()
                # print('frame:{} ValueNetLoss:{:.4f}'.format(int(frame_idx), loss.mean().item()))
                # if mean_loss < 1e-4:
                #    print('Loss too small {:.4f}'.format(mean_loss))
                #    error = True
                #    break

            if len(model.replay_buffer) > max_frames or error:
                break
            if len(model.replay_buffer) > model.reset_buffer_limit:
                model.replay_buffer.buffer_reset()
        # model.replay_buffer.buffer_reset()
        # print(count)
        # print([episode_reward.sum()/count, episode_recall.sum()/count, episode_intra.sum()/count])
        if error:
            break
        rewards.append([episode_reward / count, episode_recall.sum() / count, episode_intra.sum() / count])
        # print('buffer_size: {}'.format(len(model.replay_buffer)))
        # print('frame:{} mean_reward: R@{}:{:.4f} MRR@{}:{:.4f} IntraReward@{}:{:.4f} ValueNetLoss:{:.4f}'
        #      .format(int(frame_idx), str(cutoff), rewards[-1][1], str(cutoff), rewards[-1][0], str(cutoff), rewards[-1][2], np.mean(np.array(losses_ddpg))))

        # plot(frame_idx, rewards)
    # plot(np.arange(len(norms)), np.array(norms))

    model.save_netwokrs(model_dir, name + model.info_name)

    # print(rewards)
    return model.target_policy_net, model.target_value_net, name + model.info_name


def run_ddpg_with_reward_eval(env, model, loader, model_dir='/models', cutoff=10, name='ddpg', batch_size=32,
                              max_frames=1000000,
                              rollout=3, min_mean_reward=.3, min_steps=3, min_sessions=3, n_epochs=5):

    device = model.device
    frame_idx = 0
    rewards = []
    loader.generate_batches()
    data = loader.batches
    del loader
    epoch = 0
    while frame_idx < max_frames and epoch < n_epochs:

        losses_ddpg = []
        # This should load the same original model  restarting  users' sessions
        state = env.get_reset_state(batch_size, model.state_dim)
        state_env = env.get_reset_state(batch_size, model.state_dim)
        # This should reset the noise generator for exploration states
        model.noise_generator.reset()

        Hs, Hu = env.session_gru.init_hidden(), env.user_gru.init_hidden()
        next_H_s_init = torch.zeros(batch_size, batch_size).to(model.device)

        steps = torch.ones(batch_size, 1).to(device)
        session_count = torch.ones(batch_size, 1).to(device)
        b_steps_reward = torch.zeros(batch_size, 1).to(model.device)

        session_reward = torch.zeros(batch_size, 1).to(model.device)
        next_session_reward = torch.zeros(batch_size, 1).to(model.device)
        batch_id = 0
        total_reward = 0
        total_steps = 0
        l_total_reward = []
        acc_reward=[[]]*batch_size

        while batch_id < len(data):
            input, target, Sstart, Ustart, in_eval = data[batch_id]
            input = input.to(model.device)
            target = target.to(model.device)
            # Forward the concatenation of inputs (h_u, Embed(x) or ((h_u, h_s, Embed(x))
            # This 'action' is the new improved  hidden state which  is generated by the policy
            action = model.policy_net.get_action(state).detach()
            # Add noise to  action
            action = model.noise_generator.get_action_batch(action.detach(), steps).detach()
            # Calculate new state and reward values

            Sstart = torch.Tensor(Sstart).to(model.device).view(-1, 1)
            Ustart = torch.Tensor(Ustart).to(model.device).view(-1, 1)

            # We count the number of the steps to create an episode each k steps for long sessions
            # steps accumulates the number of intra-sessions interactions to then select the rows
            # to be stored in replay buffer
            valid_rl_rows = torch.tensor(in_eval['in_rl']).to(model.device)
            steps, session_count, mask_for_interactions_step = generate_interaction_mask(steps, session_count, Sstart,
                                                                                    Ustart, min_steps, min_sessions,
                                                                                    valid_rl_rows)

            # Performing action
            logit, Hu, Hs, new_H_s_init = env.step(input, Sstart, Ustart, Hs.detach(), Hu.detach(), action,
                                     mask_for_interactions_step, model.strategy)
            # Updating initial state for every session in batch
            H_s_init = next_H_s_init * (1.0-(Sstart+mask_for_interactions_step)) + \
                       new_H_s_init * (Sstart + mask_for_interactions_step)
            # Accumulating total frames
            frame_idx += mask_for_interactions_step.sum()
            # Calculating valid session interactions in batch
            eval_batch_slice = torch.arange(batch_size).to(model.device)[mask_for_interactions_step.view(-1).bool()]
            # Calculating reward for first step of the action
            recall, mrr, ranks = E.evaluate_with_ranks(logit[eval_batch_slice], target[eval_batch_slice],
                                                       cutoff, batch_wise=True)
            b_steps_reward[eval_batch_slice] = b_steps_reward[eval_batch_slice] + recall
            b_steps_reward_count = torch.ones(batch_size, 1).to(model.device)
            session_reward = session_reward * (1.0 - Sstart)
            session_reward[eval_batch_slice] = session_reward[eval_batch_slice] + recall

            done = torch.zeros(batch_size, 1).to(model.device)
            finishing_steps = steps
            finishing_reward = session_reward
            # Bootstrapping environment to get new state
            for b_step in range(min_steps-1):
                batch_id += 1
                if batch_id >= len(data):
                    break
                input, target, Sstart, Ustart, in_eval = data[batch_id]
                input = input.to(model.device)
                target = target.to(model.device)
                Sstart = torch.Tensor(Sstart).to(model.device).view(-1, 1)
                Ustart = torch.Tensor(Ustart).to(model.device).view(-1, 1)
                # Done condition for finished sessions
                done[Sstart.nonzero()[:, 0]] = 1.0
                finishing_steps += (1.0 - done)
                finishing_reward += b_steps_reward * (1. - done)
                valid_rl_rows = torch.tensor(in_eval['in_rl']).to(model.device)
                steps, session_count, mask_for_interactions = generate_interaction_mask(steps, session_count, Sstart,
                                                                                        Ustart, min_steps, min_sessions,
                                                                                        valid_rl_rows)

                logit, Hu, Hs , next_H_s_init= env.forward_init(input, Sstart, Ustart, Hs.detach(), Hu.detach())
                next_H_s_init = next_H_s_init * Sstart

                if torch.isnan(logit.mean()):
                    raise RuntimeError('got nan')
                # Calculating Reward fork selected rows in the batch
                recall, mrr, ranks = E.evaluate_with_ranks(logit[eval_batch_slice], target[eval_batch_slice],
                                                           cutoff, batch_wise=True)
                # Accumulating to total bootstrap reward
                b_steps_reward[eval_batch_slice] = b_steps_reward[eval_batch_slice] + recall
                b_steps_reward = (1. - Sstart) * b_steps_reward
                b_steps_reward_count = b_steps_reward_count * (1. - Sstart) + 1.0



                session_reward[eval_batch_slice] = session_reward[eval_batch_slice] + recall
                session_reward = session_reward * (1.0 - Sstart)

                # Generating Next State
                next_state = env.get_state(input, Sstart, Ustart, Hs.detach(), Hu.detach(),
                                           H_s_init.detach(),
                                           False, model.strategy, steps).to(device)

            # Calculating the mean reward of bootstrapped steps
            mean_steps_reward = b_steps_reward / b_steps_reward_count
            # Applying threshold  for low reward bootstrapped steps
            #done = done * (mean_steps_reward < min_mean_reward)
            # Accumulating whole episode reward
            total_reward += finishing_reward.sum()
            total_steps += finishing_steps.sum()
            # Push transitions to replay buffer
            model.replay_buffer.push_batch(state.detach().cpu(),
                                           action.detach().cpu(),
                                           (finishing_reward / finishing_steps).detach().cpu(),# mean_steps_reward.detach().cpu(),
                                           next_state.detach().cpu(),
                                           done.detach().cpu(),
                                           mask_for_interactions_step.flatten().detach().cpu())
            # Updating batch position
            batch_id += 1
            # Setting state for next transition
            state = next_state
            if batch_id % (len(data)//10) == 0:
                print('Completed batches: {:.2f}%({}/{}),RB_size:({}/{}){}'.format(100*batch_id/len(data),batch_id,
                                                                                    len(data),
                                                                         frame_idx, max_frames,
                                                                         len(model.replay_buffer)))
                #model.print_parameters()

            #if len(model.replay_buffer) > 10 * batch_size:
                #for i in range(5):
                #    loss = model.ddpg_update(batch_size)
                #    losses_ddpg.append(loss)
                #    if torch.isnan(loss):
                #        raise RuntimeError('value_loss nan')


            if batch_id >= len(data):
                epoch += 1
                for ind in range(30):
                    ploss,vloss = model.ddpg_update_all(batch_size)
                    #l_total_reward.append(total_reward.sum() / total_steps)
                    print(['update:'+str(ind),ploss,vloss])
                if epoch % 1 == 0:
                    print("Reseting buffer:{}".format(str(len(model.replay_buffer))))
                    model.replay_buffer.buffer_reset()
                rewards.append((total_reward/total_steps).sum())
                #print("Epoch {}: Qnetwork_loss:{}".format(epoch, torch.stack(losses_ddpg).mean().item()))
                print(rewards)
                model.save_netwokrs(model_dir, name + model.info_name+'_epoch'+str(epoch))

                break






    model.save_netwokrs(model_dir, name + model.info_name)
    print(rewards)
    return model.target_policy_net, model.target_value_net, name + model.info_name
