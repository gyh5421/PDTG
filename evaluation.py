import os
from model import DCGAN
from utils import visualize
from flowers_data import FlowersData

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("numclasses", 12, "The dimension of output [12]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("result_dir", "result", "Directory name to save the evaluation result [samples]")
flags.DEFINE_string('subset', 'train', "Either 'train' or 'validation'.")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean('log_device_placement', True, "Whether to log device placement.")
FLAGS = flags.FLAGS

def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.result_dir):
        os.makedirs(FLAGS.result_dir)
        
    dataset = FlowersData(subset=FLAGS.subset)
    assert dataset.data_files()

    dcgan = DCGAN(z_dim=200, dataset=dataset)
    dcgan.evaluate()

    if FLAGS.visualize:
        # Below is codes for visualization
        OPTION = 2
        visualize(dcgan, FLAGS, OPTION)

if __name__ == '__main__':
    tf.app.run()
