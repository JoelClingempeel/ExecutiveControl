import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from basal_ganglia import BasalGanglia

NUM_CHANNELS = 3
AUTOENCODER_CRITERION = nn.MSELoss()


class PosteriorCortex(nn.Module):
    """ Posterior cortex used for preprocessing inputs before passing to Pfc.

        Parameters:
          input_dim:  Input dimension.
          hidden_dim:  Hidden dimension.
          output_dim:  Output dimension.
          lr:  Learning used to train (as an autoencoder).
          momentum:  Momentum used to train (as an autoencoder).
          criterion:  Criterion used to measure reconstruction loss (as an autoencoder).

    """
    def __init__(self, input_dim, hidden_dim, output_dim, lr, momentum, criterion):
        super(PosteriorCortex, self).__init__()
        self.encode1 = nn.Linear(input_dim, hidden_dim)
        self.encode2 = nn.Linear(hidden_dim, output_dim)
 
        self.decode1 = nn.Linear(output_dim, hidden_dim)
        self.decode2 = nn.Linear(hidden_dim, input_dim)
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        self.criterion = AUTOENCODER_CRITERION

    def encode(self, x):
        x = F.relu(self.encode1(x))
        return F.relu(self.encode2(x))

    def decode(self, x):
        x = F.relu(self.decode1(x))
        return F.relu(self.decode2(x))
       
    def forward(self, x):
        return self.decode(self.encode(x))

    def train(self, x):
        self.optimizer.zero_grad()
        pred_x = self.forward(x)
        loss = self.criterion(pred_x, x)
        loss.backward()
        self.optimizer.step()


class Stripe(nn.Module):
    """ Stripe used for preprocessing inputs inside a Pfc layer.

        Parameters:
          input_dim:  Input dimension.
          output_dim:  Output dimension.
          batch_size:  Batch size used for training.
          lr:  Learning used to train (as an autoencoder).
          momentum:  Momentum used to train (as an autoencoder).
          criterion:  Criterion used to measure reconstruction loss (as an autoencoder).
          tensorboard_path:  Path to where tensorboard event files will be written.
          log_every_n:  Log to tensorboard the loss after every log_every_n batches.

    """
    def __init__(self, input_dim, output_dim, batch_size, lr, momentum, criterion, tensorboard_path, log_every_n):
        super(Stripe, self).__init__()
        self.encode_layer = nn.Linear(input_dim, output_dim)
        self.decode_layer = nn.Linear(output_dim, input_dim)
        self.batch_size = batch_size
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        self.criterion = criterion
        self.log_every_n = log_every_n

        self.batch = []
        self.file_writer = SummaryWriter(tensorboard_path)
        self.losses = []
        self.log_loss_count = 0

    def encode(self, x):
        return F.relu(self.encode_layer(x))

    def decode(self, x):
        return F.relu(self.decode_layer(x))
       
    def forward(self, x):
        return self.decode(self.encode(x))

    def train(self, x):
        self.batch.append(x)
        # Cache inputs, and train when cache size matches desired batch size.
        if len(self.batch) == self.batch_size:
            data = torch.stack(self.batch, dim=0)
            self.optimizer.zero_grad()
            pred_data = self.forward(data)
            loss = self.criterion(pred_data, data)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.batch = []
            self.losses.append(loss.item())

            if len(self.losses) == self.log_every_n:
                self.file_writer.add_scalar('Loss', sum(self.losses) / len(self.losses), self.log_loss_count)
                self.losses = []
                self.file_writer.flush()
                self.log_loss_count += 1


class PfcLayer:
    """Provides a Pfc Layer consisting of a set of autoencoders (called stripes)
       together with a DQN gating their activity.

       Parameters:
         stripe_class:  Class used to create stripes - can be any class implementing the Stripe API below.
         stripe_dim:  Encoding dimension of each stripe. (Assumed consistent across layer.)
         num_stripes:  Number of stripes.
         input_dim:  Dimension of input to Pfc layer across *all* stripes from the layer below.
         dqn:  DQN class object (see dqn.py).
         alpha:  Hyperparameter to control the penalty for having too many stripes active in a layer.
         batch_size:  Size of a batch used to train each stripe.
         lr:  Learning used to train (as an autoencoder).
         momentum:  Momentum used to train (as an autoencoder).
         tensorboard_path:  Path to where tensorboard event files will be written.
         log_every_n:  Log to tensorboard the loss after every log_every_n batches.
         criterion:  Criterion for measuring reconstruction loss of stripes.

    """
    def __init__(self, stripe_class, stripe_dim, num_stripes, input_dim, dqn, alpha,
                 batch_size, lr, momentum, tensorboard_path, log_every_n, criterion):
        self.stripe_dim = stripe_dim
        self.num_stripes = num_stripes
        self.input_dim = input_dim
        self.dqn = dqn
        self.alpha = alpha

        self.stripes = [stripe_class(input_dim, stripe_dim, batch_size, lr, momentum,
                                     criterion, tensorboard_path + str(j), log_every_n)
                        for j in range(num_stripes)]
        self.prev_stripe_data = torch.zeros(num_stripes, stripe_dim)
        self.stripe_data = torch.zeros(num_stripes, stripe_dim)
        self.actions = [0 for _ in range(num_stripes)]
        self.train_stripes = True

    def forward(self, data):
        data = data.reshape(-1)
        self.actions = self.dqn.select_actions(data)

        self.prev_stripe_data = torch.tensor(self.stripe_data.tolist())  # Store previous state.
        for index in range(self.num_stripes):
            if self.actions[index] == 0:  # Inactive
                self.stripe_data[index] = torch.zeros(self.stripe_dim)
            if self.actions[index] == 1:  # Read
                self.stripe_data[index] = self.stripes[index].encode(data)
                if self.train_stripes:
                    self.stripes[index].train(data)  # Train stripe (as autoencoder) on data.
            if self.actions[index] == 2:  # Maintain
                continue

        return self.stripe_data

    def reset_state(self):
        self.prev_stripe_data = torch.zeros(num_stripes, stripe_dim)
        self.stripe_data = torch.zeros(num_stripes, stripe_dim)

    def train_dqn(self, task_reward):
        num_active_stripes = len([num for num in self.actions if num == 0])
        reward = task_reward - self.alpha * num_active_stripes
        self.dqn.learn_from_experience(self.prev_stripe_data, self.actions, reward, self.stripe_data)


class Cortex:
    """Provides a cortex consisting of an autoencoder for preprocessing (posterior cortex)
       together with a collection of stripe layers.

       Parameters:
         config:  Dictionary of configs parsed from json used to create the model.
         tensorboard_path:  Path to where tensorboard event files will be written.

    """
    def __init__(self, config, tensorboard_path):
        # posterior_cortex_optimizer = optim.SGD(dqn.parameters(), lr=config['lr'], momentum=config['momentum'])
        self.posterior_cortex = PosteriorCortex(config['posterior_input_dim'],
                                                config['posterior_hidden_dim'],
                                                config['posterior_output_dim'],
                                                config['lr'],
                                                config['momentum'],
                                                AUTOENCODER_CRITERION)

        num_stripes = [1] + config['num_stripes']
        stripe_dim = [config['posterior_output_dim']] + config['stripe_dim']

        dqn_hidden_dim = 7  # TODO Get in a more systematic way.

        self.pfc_layers = []
        for index in range(1, len(num_stripes)):
            # Here the 0-th index refers to the posterior cortex output.
            input_dim = num_stripes[index - 1] * stripe_dim[index - 1]
            dqn = nn.Sequential(
                nn.Linear(input_dim, dqn_hidden_dim),
                nn.ReLU(),
                nn.Linear(dqn_hidden_dim, 3 * num_stripes[index]),
            )
            target_dqn = nn.Sequential(
                nn.Linear(input_dim, dqn_hidden_dim),
                nn.ReLU(),
                nn.Linear(dqn_hidden_dim, 3 * num_stripes[index]),
            )
            dqn_optimizer = optim.SGD(dqn.parameters(), lr=config['lr'], momentum=config['momentum'])

            dqn = BasalGanglia(dqn,
                               target_dqn,
                               dqn_optimizer,
                               num_stripes[index],
                               gamma=config['gamma'],
                               batch_size=config['batch_size'],
                               iter_before_train=config['iter_before_train'],
                               eps=config['eps'],
                               memory_buffer_size=config['memory_buffer_size'],
                               replace_target_every_n=config['replace_every_n'],
                               log_every_n=config['log_every_n'],
                               tensorboard_path=os.path.join(tensorboard_path, f'dqn_{index}'))

            pfc_layer = PfcLayer(Stripe,
                                 stripe_dim[index],
                                 num_stripes[index],
                                 input_dim,
                                 dqn,
                                 config['alpha'][index - 1],
                                 config['batch_size'],
                                 config['lr'],
                                 config['momentum'],
                                 os.path.join(tensorboard_path, f'stripe_{index}_'),
                                 config['log_every_n'],
                                 AUTOENCODER_CRITERION)
            self.pfc_layers.append(pfc_layer)

    def forward(self, data):
        data = self.posterior_cortex.encode(data)
        for stripe_layer in self.pfc_layers:
            data = stripe_layer.forward(data)
        return data

    def train_posterior_cortex(self, images, num_epochs):
        self.posterior_cortex.train(images)

    def train_basal_ganglia(self, reward):
        for layer in self.pfc_layers:
            layer.train_dqn(reward)

    def toggle_stripe_train(self, train):  # Takes boolean input.
        for layer in self.pfc_layers:
            layer.train_stripes = train
