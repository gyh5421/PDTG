import os
from model import DCGAN
from utils import visualize
from flowers_data import FlowersData

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.5]")
flags.DEFINE_integer("numclasses", 12, "The dimension of output [12]")
flags.DEFINE_integer('max_steps', 10000000, "Number of batches to run.")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("log_dir", "trainlog", "Directory name to save the log")
flags.DEFINE_string('subset', 'train', "Either 'train' or 'validation'.")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean('log_device_placement', True, "Whether to log device placement.")
FLAGS = flags.FLAGS

def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
        
    dataset = FlowersData(subset=FLAGS.subset)
    assert dataset.data_files()

    dcgan = DCGAN(z_dim=200, dataset=dataset)
    dcgan.train()

    if FLAGS.visualize:
        # Below is codes for visualization
        OPTION = 2
        visualize(dcgan, FLAGS, OPTION)

if __name__ == '__main__':
    tf.app.run()
