import torch
import torch.nn as nn
import torch.nn.functional as F

from basal_ganglia import BasalGanglia


# TODO Add convolutional layer(s).
class PosteriorCortex(nn.Module):
    """ Posterior cortex used for preprocessing inputs before passing to Pfc.

        Parameters:
          input_dim:  Input dimension
          output_dim:  Output dimension.
          optimizer:  Optimizer object used to train (as an autoencoder).
          criterion:  Criterion used to measure reconstruction loss (as an autoencoder).

    """
    def __init__(self, input_dim, output_dim, optimizer, criterion):
        super(Stripe, self).__init__()
        self.encode = nn.Linear(input_dim, output_dim)
        self.decode = nn.Linear(output_dim, input_dim)
        self.optimizer = optimizer
        self.criterion = criterion

    def encode(self, x):
        return F.relu(self.encode(x))

    def decode(self, x):
        return F.relu(self.decode(x))
       
    def forward(self, x):
        return self.decode(self.encode(x))

    def train(self, x):  # TODO Implement logging system.
        optimizer.zero_grad()
        pred_x = self.forward(x)
        loss = criterion(pred_x, x)
        loss.backward()
        optimizer.step()


class Stripe(nn.Module):
    """ Stripe used for preprocessing inputs inside a Pfc layer.

        Parameters:
          input_dim:  Input dimension
          output_dim:  Output dimension.
          batch_size:  Batch size used for training.
          optimizer:  Optimizer object used to train (as an autoencoder).
          criterion:  Criterion used to measure reconstruction loss (as an autoencoder).

    """
    def __init__(self, input_dim, output_dim, batch_size, optimizer, criterion):
        super(Stripe, self).__init__()
        self.encode = nn.Linear(input_dim, output_dim)
        self.decode = nn.Linear(output_dim, input_dim)
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.criterion = criterion

        self.batch = []

    def encode(self, x):
        return F.relu(self.encode(x))

    def decode(self, x):
        return F.relu(self.decode(x))
       
    def forward(self, x):
        return self.decode(self.encode(x))

    def train(self, x):  # TODO Implement logging system.
        self.batch.append(x)
        # Cache inputs, and train when cache size matches desired batch size.
        if len(self.batch) == self.batch_size:
            data = torch.stack(self.batch, dim=0)
            optimizer.zero_grad()
            pred_data = self.forward(data)
            loss = criterion(pred_data, data)
            loss.backward()
            optimizer.step()
            self.batch = []


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

    """
    def __init__(self, stripe_class, stripe_dim, num_stripes, dqn, alpha, batch_size):
        self.stripe_dim = stripe_dim
        self.num_stripes = num_stripes
        self.input_dim = input_dim
        self.dqn = dqn
        self.alpha = alpha

        # TODO get input_dim
        self.stripes = [stripe_class(input_dim, stripe_dim, batch_size, optimizer, criterion)
                        for _ in range(num_stripes)]
        self.prev_stripe_data = torch.zeros(num_stripes, stripe_dim)
        self.stripe_data = torch.zeros(num_stripes, stripe_dim)
        self.actions = [0 for _ in range(num_stripes)]

    def forward(self, data):
        self.actions = self.dqn.select_actions(data)

        self.prev_stripe_data = torch.tensor(self.stripe_data.tolist())  # Store previous state.
        for index in range(self.num_stripes):
            if self.actions[index] == 0:  # Inactive
                self.stripe_data[index] = torch.zeros(self.stripe_dim)
            if self.actions[index] == 1:  # Read
                self.stripe_data[index] = self.stripe[index].encode(data)
                self.stripe[index].train(data)  # Train stripe (as autoencoder) on data.
            if self.actions[index] == 2:  # Maintain
                continue

        return torch.cat(self.stripe_data)

    def reset_state(self):
        self.prev_stripe_data = torch.zeros(num_stripes, stripe_dim)
        self.stripe_data = torch.zeros(num_stripes, stripe_dim)

    def train_dqn(self, task_reward):
        num_active_stripes = len([num for num in self.actions if num == 0])
        reward = task_reward - self.alpha * num_active_stripes
        # TODO Verify if the below step should require calling .unsqueeze(0) on prev/curr stripe data.
        dqn.learn_from_experience(self.prev_stripe_data, self.action, reward, stripe_data)


class Cortex:
    """Provides a cortex consisting of an autoencoder for preprocessing (posterior cortex)
       together with a collection of stripe layers.

       Parameters:
         config:  Dictionary of configs parsed from json used to create the model.
         tensorboard_path:  Path to where tensorboard event files will be written.

    """
    def __init__(self, config, tensorboard_path):
        # TODO What input vals should be saved?
        self.posterior_cortex = PosteriorCortex()
        num_stripes = [1] + config['num_stripes']
        stripe_dim = [posterior_output_dim] + config['stripe_dim']
        for index in range(1, len(num_stripes)):
          # Here the 0-th index refers to the posterior cortex output.
           dqn = nn.Sequential(
                nn.Linear(num_stripes[index - 1] * stripe_dim[index - 1], dqn_hidden_dim),
                nn.ReLU(),
                nn.Linear(dqn_hidden_dim, 3 * num_stripes[index]),
            )
            target_dqn = nn.Sequential(
                nn.Linear(num_stripes[index - 1] * stripe_dim[index - 1], dqn_hidden_dim),
                nn.ReLU(),
                nn.Linear(dqn_hidden_dim, 3 * num_stripes[index]),
            )
            dqn_optimizer = optim.SGD(dqn.parameters(), lr=config['lr'], momentum=config['momentum'])
            dqn = BasalGanglia(dqn,
                               target_dqn,
                               dqn_optimizer,
                               num_heads,
                               actions_per_head,
                               gamma=config['gamma'],
                               batch_size=config['batch_size'],
                               iter_before_train=config['iter_before_train'],
                               eps=config['eps'],
                               memory_buffer_size=config['memory_buffer_size'],
                               replace_every_n=config['replace_every_n'],
                               log_every_n=config['log_every_n'],
                               tensorboard_path=tensorboard_path)  # TODO Join with subdir.

            pfc_layer = PfcLayer(Stripe, stripe_dim[index], num_stripes[index], dqn, config['alpha'], config['batch_size'])
            self.pfc_layers.append(pfc_layer)

    def forward(self, data):
        data = self.posterior_cortex(data)
        for stripe_layer in self.stripe_layers:
            data = stripe_layer.forward(data)
        return data
