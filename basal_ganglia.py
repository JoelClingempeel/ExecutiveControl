import random

import torch
from torch.utils.tensorboard import SummaryWriter

NUM_ACTIONS = 3


class BasalGanglia:
    """
      Parameters:
        q_net:  Network used to predict q-values.
        target_q_net:  Used to avoid too much "chasing a moving target" - typically duplicate of above.
        optimizer:  Optimizer.
        num_stripes:  Number of independent selections to be made (one per stripe).
        gamma:  Discounting reward factor in Bellman equation.
        batch_size  Batch size.
        iter_before_train:  Number of iterations used to expand memory buffer before training may begin.
        eps:  Probability of making random choices instead of 
        memory_buffer_size:  Maximum size of memory replay buffer.
        replace_target_every_n:  Frequency with which q_net's parameters are copied to target_q_net.
        log_every_n:  Log after every n batches.
        tensorboard_path:  Directory used for tensorboard logging.
    
    """
    def __init__(self, q_net, target_q_net, optimizer, num_stripes=7, gamma=.3, batch_size=8,
                 iter_before_train=50, eps=.1, memory_buffer_size=100, replace_target_every_n=100,
                 log_every_n=100, tensorboard_path='logs'):
        self.q_net = q_net
        self.target_q_net = target_q_net
        self.optimizer = optimizer
        self.num_stripes = num_stripes
        self.gamma = gamma
        self.batch_size = batch_size
        self.iter_before_train = iter_before_train
        # TODO Consider each head *independently* being epsilon-greedy.
        self.eps = eps
        self.memory_buffer_size = memory_buffer_size
        self.replace_target_every_n = replace_target_every_n
        self.log_every_n = log_every_n
        self.writer = SummaryWriter(tensorboard_path)
        self.memory_buffer = []
        self.losses = []
        self.rewards = []
        self.loss_log_count = 0
        self.reward_log_count = 0

    def get_q_values(self, state, use_target=False):
        if use_target:
            return self.target_q_net(q_net_input)
        else:
            return self.q_net(q_net_input)

    def select_actions(self, state):
        if (len(self.memory_buffer) < self.iter_before_train or
                random.uniform(0, 1) < self.eps):
            return [random.randint(0, NUM_ACTIONS)
                    for _ in range(self.num_stripes)]
        else:
            q_vals = self.get_q_values(state.unsqueeze(0))
            q_vals = q_vals.reshape(self.num_stripes, NUM_ACTIONS)
            return torch.argmax(q_vals, dim=1).tolist()

    def train_iterate(self):
        samples = random.sample(self.memory_buffer, self.batch_size)
        loss = torch.tensor(0)
        optimizer.zero_grad()

        for state, action, reward, new_state in samples:
            current_q_values = self.get_q_values(state).squeeze(0).reshape(-1, 3)
            current_q_value = torch.tensor(0)
            for q_values, subaction in zip(current_q_values, action):
                current_q_value += q_values[action]

            future_q_values = self.get_q_values(new_state, use_target=True).detach().squeeze(0).reshape(-1, 3)
            future_q_value = torch.sum(torch.max(future_q_values, dim=1))  # TODO new_state or state?
            

            loss += (current_q_value - (reward + self.gamma * future_q_value)) ** 2

        loss.backward(retain_graph=True)
        optimizer.step()
        self.losses.append(loss.item())

    def learn_from_experience(self, prev_state, action, reward, curr_state):
        self.rewards.append(reward)

        # Update memory buffer and (if applicable) train.
        self.memory_buffer.append([prev_state, action, reward, curr_state])
        if len(self.memory_buffer) > self.memory_buffer_size:
            self.memory_buffer.pop(0)
        if len(self.memory_buffer) >= self.iter_before_train:
            self.train_iterate()

        if len(self.losses) == self.log_every_n:
            self.writer.add_scalar('Loss', sum(self.losses) / len(self.losses), self.loss_log_count)
            self.losses = []
            self.writer.flush()
            self.loss_log_count += 1

        if len(self.rewards) == self.log_every_n:
            self.writer.add_scalar('Reward', sum(self.rewards) / len(self.rewards), self.reward_log_count)
            self.rewards = []
            self.writer.flush()
            self.reward_log_count += 1

            if (iteration + 1) % self.replace_target_every_n == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())
