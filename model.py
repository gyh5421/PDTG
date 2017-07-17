import os
import csv
from datetime import datetime
import time
from scipy import stats
import numpy as np
import tensorflow as tf
import inception_model as inception

from utils import save_images

import image_processing
import slim.ops
import slim.scopes

BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999

FLAGS = tf.app.flags.FLAGS

class DCGAN(object):
  def __init__(self,z_dim,dataset):
    self.z_dim=z_dim
    self.dataset=dataset
  def build_model(self,batch_size,images,labels,numclasses,is_training,restore):

    self.z = tf.random_uniform([batch_size,self.z_dim], -1, 1, dtype=tf.float32)
    
    tf.histogram_summary('z', self.z)
    
    self.G = self.generator(y=labels, y_dim=numclasses, is_training=is_training, restore=restore, scope='g')
    self.D = self.discriminator(images, labels, reuse=False, is_training=is_training, restore=restore, scope='d')

    self.D_ = self.discriminator(self.G, labels, reuse=True, is_training=is_training, restore=restore, scope='d')

    self.G_sum = tf.image_summary("G", self.G, name='g/image')

    self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_, tf.zeros_like(self.D_)))
    self.g_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_, tf.ones_like(self.D_)))
    self.g_loss_h = self.perception_loss(self.G, labels, numclasses, False, True, 'h')
    
    self.g_loss = self.g_loss_d + 10*self.g_loss_h
    
#    self.g_regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='g')
#    self.g_total_loss = tf.add_n([self.g_loss] + self.g_regularization_losses, name='g_total_loss')
      
    self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real, name='d/loss_real')
    self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake, name='d/loss_fake')
                                                
    self.d_loss = 0.5*self.d_loss_real + 0.5*self.d_loss_fake
    
#    self.d_regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='d')
#    self.d_total_loss = tf.add_n([self.d_loss] + self.d_regularization_losses, name='d_total_loss')

    self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss, name='g/loss')
    self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss, name='d/loss')

    t_vars = tf.trainable_variables()
    

    self.d_vars = [var for var in t_vars if 'd/' in var.name]
    self.g_vars = [var for var in t_vars if 'g/' in var.name]   
  def train(self):
    """Train DCGAN"""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
      # Override the number of preprocessing threads to account for the increased
      # number of GPU towers.
      num_preprocess_threads = FLAGS.num_preprocess_threads
      images, labels = image_processing.distorted_inputs(self.dataset, num_preprocess_threads=num_preprocess_threads)
  
      with tf.device('/gpu:0'):
        # Set weight_decay for weights in Conv and FC layers.
        
        self.build_model(FLAGS.batch_size, images, labels, 12, True, False)
            
        d_opt = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                      .minimize(self.d_loss, var_list=self.d_vars)
        g_opt = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                      .minimize(self.g_loss, var_list=self.g_vars)
                      
        train_op = tf.group(d_opt, g_opt, g_opt)

        batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)

      # Add a summaries for the input processing and global_step.
      summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

      # Group all updates to into a single train op.
      batchnorm_updates_op = tf.group(*batchnorm_updates)
      train_op = tf.group(train_op, batchnorm_updates_op)
  
      # Create a saver.
      saver = tf.train.Saver(tf.all_variables())
  
      summary_op = tf.merge_summary(summaries)
  
      # Build an initialization operation to run below.
      init = tf.initialize_all_variables()
  
      # Start running operations on the Graph. allow_soft_placement must be set to
      # True to build towers on GPU, as some of the ops do not have GPU
      # implementations.
      sess = tf.Session(config=tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=FLAGS.log_device_placement))
      sess.run(init)
  
  
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        variables_to_restore = tf.get_collection(
            slim.variables.VARIABLES_TO_RESTORE)
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, ckpt.model_checkpoint_path)
        print('%s: Pre-trained model restored from %s' %
              (datetime.now(), FLAGS.checkpoint_dir))
  
      # Start the queue runners.
      tf.train.start_queue_runners(sess=sess)
      summary_writer = tf.train.SummaryWriter(
          FLAGS.log_dir,
          graph=sess.graph)
  
      for step in xrange(FLAGS.max_steps):
        start_time = time.time()
        sess.run([train_op])
        duration = time.time() - start_time
  
        if step % 10 == 0:
          examples_per_sec = FLAGS.batch_size / float(duration)
          format_str = ('%s: step %d(%.1f examples/sec; %.3f '
                        'sec/batch)')
          print(format_str % (datetime.now(), step, examples_per_sec, duration))
  
        if step % 100 == 0:
          summary_str = sess.run(summary_op)
          summary_writer.add_summary(summary_str, step)
          samples = sess.run(self.G)
          save_images(samples, './%s/%d' % (FLAGS.sample_dir, step))
  
        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
          checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)
  def evaluate(self):
    """Evaluate DCGAN"""
    csvfile=file('ave_20.csv','rb')
    unique_labels = np.array([[float(e) for e in l] for l in csv.reader(csvfile)])
    csvfile.close()
    unique_labels = unique_labels*0.9
#    unique_labels = stats.zscore(unique_labels)
#    unique_labels = np.array([[min(max(e,-3),3) for e in l] for l in unique_labels])*0.3
    with tf.Graph().as_default():
      with tf.device('/cpu:0'):
        batch_norm_params = {'decay': BATCHNORM_MOVING_AVERAGE_DECAY,'epsilon': 1e-5}
        # Set weight_decay for weights in Conv and FC layers.
        with slim.scopes.arg_scope([slim.ops.conv2d, slim.ops.deconv2d, slim.ops.fc], 
                            stddev=0.02, 
                            activation=tf.nn.relu, 
                            batch_norm_params=batch_norm_params,
                            weight_decay=0):
                              
          with slim.scopes.arg_scope([slim.ops.conv2d, slim.ops.deconv2d, slim.ops.fc, slim.ops.batch_norm, slim.ops.dropout],
                    is_training=False):

            self.z = tf.placeholder(tf.float32, (FLAGS.batch_size, self.z_dim))
            labels = tf.placeholder(tf.float32, (FLAGS.batch_size, 12))
            self.G = self.generator(y=labels, y_dim=12, is_training=False, restore=True, scope='g')
      
  
      # Build an initialization operation to run below.
      init = tf.initialize_all_variables()
  
      # Start running operations on the Graph. allow_soft_placement must be set to
      # True to build towers on GPU, as some of the ops do not have GPU
      # implementations.
      sess = tf.Session(config=tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=FLAGS.log_device_placement))
      sess.run(init)
  
  
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
#        ema = tf.train.ExponentialMovingAverage(BATCHNORM_MOVING_AVERAGE_DECAY)
#        variables_to_restore = ema.variables_to_restore()
        restorer = tf.train.Saver(tf.all_variables())
        restorer.restore(sess, ckpt.model_checkpoint_path)
        print('%s: Pre-trained model restored from %s' %
              (datetime.now(), FLAGS.checkpoint_dir))
      k = 1
      for l in unique_labels:
	label=np.tile(l, (FLAGS.batch_size,1))
#	for i in range(101):
#	  label[i,8]=1-i*0.02
        samples = sess.run(self.G, feed_dict={labels:label, self.z:np.random.uniform(-1,1,(FLAGS.batch_size,self.z_dim))})
        save_images(samples, './%s/%d' % (FLAGS.result_dir, k))
        k = k+1
        
  def discriminator(self, image, y, reuse, is_training, restore, scope):
    batch_norm_params = {'decay': BATCHNORM_MOVING_AVERAGE_DECAY,'epsilon': 0.001}
    with tf.variable_op_scope([image], scope, 'd', reuse=reuse):
      with slim.scopes.arg_scope([slim.ops.conv2d, slim.ops.fc],
                stddev=0.1, 
                activation=tf.nn.relu, 
                batch_norm_params=batch_norm_params,
                weight_decay=0.00000,
                is_training=is_training, 
                restore=restore):
        with slim.scopes.arg_scope([slim.ops.conv2d, slim.ops.max_pool, slim.ops.avg_pool],
                          stride=1, padding='VALID'):
          # 299 x 299 x 1
          h0 = slim.ops.conv2d(image, 32, [5, 5], stride=2, stddev=0.1, scope='conv0')
          # 148 x 148 x 32
          h1 = slim.ops.conv2d(h0, 64, [5, 5], stride=2, stddev=0.071, padding='SAME', scope='conv1')
          # 74 x 74 x 64
          h2 = slim.ops.conv2d(h1, 128, [5, 5], stride=2, stddev=0.05, padding='SAME', scope='conv2')
          # 37 x 37 x 128
  #        h2 = slim.ops.max_pool(h2, [3, 3], stride=2, scope='pool1')
          # 73 x 73 x 64
          h3 = slim.ops.conv2d(h2, 256, [5, 5], stride=2, stddev=0.035, scope='conv3')
          # 17 x 17 x 256.
          h4 = slim.ops.conv2d(h3, 512, [5, 5], stride=2, stddev=0.025, scope='conv4')
          # 7 x 7 x 512.
  #        h4 = slim.ops.max_pool(h4, [3, 3], stride=2, scope='pool2')
          # 35 x 35 x 192.
          h5 = slim.ops.conv2d(h4, 1024, [5, 5], stride=2, stddev=0.029, scope='conv5')
          # 2 x 2 x 1024.
          h5 = slim.ops.avg_pool(h5, [2, 2])
          # 1 x 1 x 256
  #        h6 = slim.ops.conv2d(h5, 768, [1, 1], scope='proj3')
          # 17 x 17 x 768.
  #        h7 = slim.ops.conv2d(h6, 1280, [3, 3], stride=2, scope='conv4')
          # 8 x 8 x 1280.
  #        h8 = slim.ops.conv2d(h7, 2048, [1, 1], scope='proj4')
          # 8 x 8 x 2048.
          
  #        shape = h8.get_shape()
  #        h9 = slim.ops.avg_pool(h8, shape[1:3], padding='VALID', scope='pool3')
          # 1 x 1 x 2048.
          h6 = slim.ops.flatten(h5)

	  h6 = slim.ops.fc(h6, 100, stddev=0.14)
        
          y = slim.ops.fc(y, 100, stddev=0.29)
        
          h6 = tf.concat(1, [h6, y])
        
          h6 = slim.ops.fc(h6, 100, stddev=0.14)
         
          h7 = slim.ops.fc(h6, 1, activation=None, stddev=4, batch_norm_params=None, scope='fc1')
  
          return h7
        
  def generator(self, y, y_dim, is_training, restore, scope):
    batch_norm_params = {'decay': BATCHNORM_MOVING_AVERAGE_DECAY,'epsilon': 0.001}
    with tf.variable_op_scope([self.z], scope, 'g'):
      with slim.scopes.arg_scope([slim.ops.deconv2d, slim.ops.conv2d, slim.ops.fc],
                stddev=0.1, 
                activation=tf.nn.relu, 
                batch_norm_params=batch_norm_params,
                weight_decay=0.00000,
                is_training=is_training, 
                restore=restore):
        with slim.scopes.arg_scope([slim.ops.deconv2d, slim.ops.conv2d], stride=2, padding='SAME'):
          # project `z` and reshape
          z_dim = self.z_dim
          if not y is None:
            y = slim.ops.fc(y, 800, stddev=0.29)
            z = tf.concat(1, [self.z, y])
            z_dim = z_dim+800
          
              # project `z` and reshape
            h0 = slim.ops.deconv2d(tf.reshape(z, [FLAGS.batch_size, 1, 1, z_dim]), [FLAGS.batch_size, 5, 5, 1024], [5, 5], padding='VALID', stddev=0.0088, scope='deconv1')
            
    #        h1 = slim.ops.deconv2d(tf.reshape(h0, [FLAGS.batch_size, 1, 1, 1024]), [FLAGS.batch_size, 3, 3, 256], [3, 3], padding='VALID', scope='deconv0')
      
    #        h2 = slim.ops.conv2d(h1, 512, [1, 1], scope='proj1')
            
    #        h1 = slim.ops.deconv2d(h0, [FLAGS.batch_size, 5, 5, 512], [5, 5], stride=2, scope='deconv1')
            
    #        h4 = slim.ops.conv2d(h3, 512, [1, 1], scope='proj2')
            
    #        h1 = slim.ops.deconv2d(h0, [FLAGS.batch_size, 5, 5, 512], [5, 5], padding='VALID', scope='deconv1')
            
            h1 = slim.ops.deconv2d(h0, [FLAGS.batch_size, 10, 10, 512], [5, 5], stride=2, stddev=0.0125, scope='deconv2')
            
    #        h6 = slim.ops.conv2d(h5, 512, [1, 1], scope='proj3')
    
            h2 = slim.ops.deconv2d(h1, [FLAGS.batch_size, 19, 19, 256], [5, 5], stride=2, stddev=0.018, scope='deconv3')
            
    #        h6 = slim.ops.conv2d(h5, 128, [1, 1], scope='proj2')
    
            h3 = slim.ops.deconv2d(h2, [FLAGS.batch_size, 38, 38, 128], [5, 5], stride=2, stddev=0.025, scope='deconv4')
    
            h4 = slim.ops.deconv2d(h3, [FLAGS.batch_size, 75, 75, 64], [5, 5], stride=2, stddev=0.035, scope='deconv5')
            
    #        h9 = slim.ops.conv2d(h8, 64, [1, 1], scope='proj4')
            
            h5 = slim.ops.deconv2d(h4, [FLAGS.batch_size, 150, 150, 32], [5, 5], stride=2, stddev=0.05, scope='deconv6')
            
    #        h11 = slim.ops.conv2d(h10, 32, [1, 1], scope='proj4')
    
            h6 = slim.ops.deconv2d(h5, [FLAGS.batch_size, 299, 299, 1], [5, 5], stride=2, stddev=0.2, scope='deconv7', activation=tf.nn.tanh, batch_norm_params=None)
    
            return h6

  def perception_loss(self, images, labels, num_classes, is_training, restore, scope):
    """Calculate the total loss on a single tower running the ImageNet model.
  
    We perform 'batch splitting'. This means that we cut up a batch across
    multiple GPU's. For instance, if the batch size = 32 and num_gpus = 2,
    then each tower will operate on an batch of 16 images.
  
    Args:
      images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                         FLAGS.image_size, 3].
      labels: 1-D integer Tensor of [batch_size].
      num_classes: number of classes
      scope: unique prefix string identifying the ImageNet tower, e.g.
        'tower_0'.
  
    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """
    # When fine-tuning a model, we do not restore the logits but instead we
    # randomly initialize the logits. The number of classes in the output of the
    # logit is the number of classes in specified Dataset.
    
    # Build inference Graph.
    with tf.name_scope(scope) as scope:
      with tf.device('/gpu:0'): 
            
    	 logits = inception.inference(tf.tile(images,[1,1,1,3]), num_classes, for_training=is_training, restore_logits=restore, scope=scope)
    	  
    	 # Build the portion of the Graph calculating the losses. Note that we will
    	 # assemble the total_loss using a custom function below.
    	 split_batch_size = images.get_shape().as_list()[0]
    	 inception.loss(logits, labels, batch_size=split_batch_size)
    	  
    	 # Assemble all of the losses for the current tower only.
    	 losses = tf.get_collection(slim.losses.LOSSES_COLLECTION, scope)
    	  
    	 # Calculate the total loss for the current tower.
    	 total_loss = tf.add_n(losses, name='total_loss')
    	  
    	    
    	 # Attach a scalar summmary to all individual losses and the total loss; do the
    	 # same for the averaged version of the losses.
    	 for l in losses:
    	   # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    	   # session. This helps the clarity of presentation on TensorBoard.
    	   loss_name = l.op.name
    	   # Name each loss as '(raw)' and name the moving average version of the loss
    	   # as the original loss name.
    	   tf.scalar_summary(loss_name, l)
    	  
    	 return total_loss                        

  
  


  
