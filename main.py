import argparse
import pickle
import yaml

import torch
from torch.utils.data import DataLoader

from cortex import Cortex

parser = argparse.ArgumentParser()

parser.add_argument('--config_file', type=str, default='config.json')
parser.add_argument('--images_path', type=str, default='data/images')
parser.add_argument('--labels_path', type=str, default='data/labels')
parser.add_argument('--tensorboard_path', type=str, default='tboard')

args = vars(parser.parse_args())


def get_data(images_path, labels_path, batch_size):
    with open(images_path, 'rb') as f:
        images = pickle.load(f)

    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)

    return DataLoader(list(zip(images, labels)),
                      batch_size=batch_size,
                      shuffle=True)


def pretrain_posterior_cortex(cortex, images_path, labels_path, batch_size, num_epochs):
    dataset = get_data(images_path, labels_path, batch_size)
    for _ in range(num_epochs):
        for images, labels in dataset:
            cortex.train_posterior_cortex(images, num_epochs)


def score_task(cortex_output, label, num_colors):
    # TODO Generalize to allow for tasks with a continuous component.
    num_colors = 9

    avg_activations = torch.mean(cortex_output, dim=1)
    # Here the first num_colors output stripes encode the color.
    # The rest encode the number of sides.
    pred_color = torch.argmax(avg_activations[:num_colors]).item()
    pred_num_sides = torch.argmax(avg_activations[num_colors:]).item() + 3    

    color = label[0, 0].item()
    num_sides = label[0, 1].item() 

    reward = 0
    if pred_color == color:
        reward += 1
    if pred_num_sides == num_sides:
        reward += 1
    return reward

def train_cortex(cortex, images_path, labels_path, num_colors):
    # TODO Vectorize cortex operations so the batch size doesn't need to be set to 1.
    dataset = get_data(args['images_path'], args['labels_path'], 1)
    for image, label in dataset:
        cortex_output = cortex.forward(image)
        reward = score_task(cortex_output, label, num_colors)
        cortex.train_basal_ganglia(reward)


def main(args):
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    cortex = Cortex(config, args['tensorboard_path'])
    pretrain_posterior_cortex(cortex, args['images_path'], args['labels_path'], config['batch_size'],
                              config['num_pretrain_epochs'])
    train_cortex(cortex, args['images_path'], args['labels_path'], config['num_colors'])


if __name__ == '__main__':
    main(args)

