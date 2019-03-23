import sys
sys.path.append('..')
import os
import yaml
import argparse
from shutil import copyfile
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.framework.ops import reset_default_graph

from src import engine, datagen, utils

def main(config_file):

    reset_default_graph()
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
    data = datagen.Data(**config['datagen'])
    train_iter, val_iter = data.datagen(output_dir+'/logs/data', **config['datagen'])
    
    # Build the model
    model = engine.Engine(**config['model'])
    model.train(train_iter, val_iter, output_dir, **config['train'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="configuration Filepath", type=str)
    args = parser.parse_args()
    main(args.config)
