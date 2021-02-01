import argparse
import pickle
from torch.utils.data import DataLoader
import yaml

# from cortex import Cortex

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
            cortex.train_posterior_cortex(images)


def train_cortex(cortex, images_path, labels_path):
    # TODO Vectorize cortex operations so the batch size doesn't need to be set to 1.
    dataset = get_data(args['images_path'], args['labels_path'], 1)
    for image, label in dataset:
        cortex_output = cortex.forward(image)
        # The predicted layer is the stripe from the final layer with the highest average activation.
        pred_label = torch.argmax(torch.mean(cortex_output, dim=1)).item()
        # TODO For tasks with a continuous element, need to adjust reward below.
        reward = 1 if pred_label == label else 0
        cortex.train_basal_ganglia(reward)


def main(args):
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    cortex = Cortex(config, args['tensorboard_path'])
    pretrain_posterior_cortex(cortex, args['images_path'], args['labels_path'], config['batcH_size'],
                              config['num_pretrain_epochs'])
    train_cortex(cortex, args['images_path'], args['labels_path'])


if __name__ == '__main__':
    main(args)

