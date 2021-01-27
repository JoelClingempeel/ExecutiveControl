import argparse
import json

from cortex import Cortex

parser = argparse.ArgumentParser()

parser.add_argument('--config_file', type=str, default='config.json')
parser.add_argument('--tensorboard_path', type=str, default='tboard')

args = vars(parser.parse_args())


def main(args):
    with open(args['config_file']) as f:
        config = json.loads(f.read())

    cortex = Cortex(config, args['tensorboard_path'])


if __name__ == '__main__':
    main(args)
