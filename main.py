import argparse
import datetime
import os
import pickle
import yaml

import torch
from torch.utils.data import DataLoader

from cortex import Cortex

parser = argparse.ArgumentParser()

parser.add_argument('--config_file', type=str, default='config.yaml')
parser.add_argument('--images_path', type=str, default='data/images')
parser.add_argument('--labels_path', type=str, default='data/labels')
parser.add_argument('--tensorboard_path', type=str, default='tboard')
parser.add_argument('--max_images', type=int, default=100)  # For debugging - otherwise set to dataset size.

args = vars(parser.parse_args())



def get_data(images_path, labels_path, batch_size, max_images):
    with open(images_path, 'rb') as f:
        images = pickle.load(f)
        images = images[:max_images]
        images = images.reshape(len(images), -1)  # Flatten images. 

    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)

    return DataLoader(list(zip(images, labels)),
                      batch_size=batch_size,
                      shuffle=True)


def pretrain_posterior_cortex(cortex, images_path, labels_path, batch_size, num_epochs, max_images):
    dataset = get_data(images_path, labels_path, batch_size, max_images)
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


def train_cortex(cortex, images_path, labels_path, num_colors, max_images):
    # TODO Vectorize cortex operations so the batch size doesn't need to be set to 1.
    dataset = get_data(args['images_path'], args['labels_path'], 1, max_images)
    for image, label in dataset:
        cortex_output = cortex.forward(image)
        reward = score_task(cortex_output, label, num_colors)
        cortex.train_basal_ganglia(reward)


def main(args):
    with open(args['config_file']) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    timestamp = str(datetime.datetime.now()).replace(' ', '_')
    logging_path = os.path.join(args['tensorboard_path'], timestamp)
    print(f'Now logging results to {logging_path}')

    cortex = Cortex(config, logging_path)
    pretrain_posterior_cortex(cortex, args['images_path'], args['labels_path'], config['batch_size'],
                              config['num_pretrain_epochs'], args['max_images'])
    train_cortex(cortex, args['images_path'], args['labels_path'], config['num_colors'], args['max_images'])


if __name__ == '__main__':
    main(args)
