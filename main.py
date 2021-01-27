import argparse

import torch

from model import *

parser = argparse.ArgumentParser()

parser.add_argument('--stripe_dim', type=int, default=5)

args = vars(parser.parse_args())


def main(args):
    for _ in range(num_pfc_layers):
        dqn = nn.Sequential(
            nn.Linear(num_symbols * 3 + 3, dqn_hidden_dim),
            nn.ReLU(),
            nn.Linear(dqn_hidden_dim, 4),
        )
        target_dqn = nn.Sequential(
            nn.Linear(num_symbols * 3 + 3, dqn_hidden_dim),
            nn.ReLU(),
            nn.Linear(dqn_hidden_dim, 4),
        )
        dqn_optimizer = optim.SGD(dqn.parameters(), lr=args['lr'], momentum=args['momentum'])
        dqn = DQN(dqn,
                  target_dqn,
                  dqn_optimizer,
                  num_heads,
                  actions_per_head,
                  gamma=args['gamma'],
                  batch_size=args['batch_size'],
                  iter_before_train=args['iter_before_train'],
                  eps=args['eps'],
                  memory_buffer_size=args['memory_buffer_size'],
                  replace_every_n=args['replace_every_n'],
                  log_every_n=args['log_every_n'],
                  tensorboard_path=args['tensorboard_path'])


if __name__ == '__main__':
    main(args)
