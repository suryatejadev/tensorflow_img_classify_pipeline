import sys
sys.path.append('..')
import os
import yaml
import argparse
from shutil import copyfile
import tensorflow

from src import engine, datagen, utils

def main(config_file):
    # parser config
    with open(config_file) as f:
        config = yaml.load(f)

    # output dir
    output_dir = os.path.join(config['output_dir'], config['exp_id'])
    utils.create_dirs([output_dir, \
            output_dir + '/checkpoints', \
            output_dir + '/logs', \
            output_dir + '/results'])
    copyfile(config_file, os.path.join(output_dir, 'config.yaml'))

    # Get train and val datagen
    data = datagen.Data(**config['data'])
    train_iter, val_iter = data.get_datagen(**config['datagen'])

    # Build the model
    model = engine.Engine(config['model'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="configuration Filepath", type=str)
    args = parser.parse_args()
    main(args.config)
