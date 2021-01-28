import argparse
import yaml

from cortex import Cortex

parser = argparse.ArgumentParser()

parser.add_argument('--config_file', type=str, default='config.json')
parser.add_argument('--tensorboard_path', type=str, default='tboard')

args = vars(parser.parse_args())


def main(args):
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    cortex = Cortex(config, args['tensorboard_path'])


if __name__ == '__main__':
    main(args)
