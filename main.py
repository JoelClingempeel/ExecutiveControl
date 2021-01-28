import argparse
import pickle
import yaml

from cortex import Cortex

parser = argparse.ArgumentParser()

parser.add_argument('--config_file', type=str, default='config.json')
parser.add_argument('--images_path', type=str, default='data/images')
parser.add_argument('--labels_path', type=str, default='data/labels')
parser.add_argument('--tensorboard_path', type=str, default='tboard')

args = vars(parser.parse_args())


def main(args):
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open(args['images_path'], 'rb') as f:
        images = pickle.load(f)

    with open(args['labels_path'], 'rb') as f:
        labels = pickle.load(f)

    cortex = Cortex(config, args['tensorboard_path'])


if __name__ == '__main__':
    main(args)
