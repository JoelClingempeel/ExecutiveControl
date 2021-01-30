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



def main(args):
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset = get_data(args['images_path'], args['labels_path'], config['batch_size'])

    for image, label in dataset:
        print(image.shape)
        print(label)

    # cortex = Cortex(config, args['tensorboard_path'])


if __name__ == '__main__':
    main(args)
