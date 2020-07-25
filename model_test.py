from utils import ( 
  imsave,
  prepare_data
)

import time
import os
import matplotlib.pyplot as plt
import re
import numpy as np
import tensorflow as tf
import scipy.io as scio
from ops import *



class T_CNN(object):

  def __init__(self,
               sess,
               image_height=224,
               image_width=224,
               label_height=224,
               label_width=224,
               batch_size=1,
               c_dim=3, 
               c_depth_dim=1,
               checkpoint_dir=None,
               ):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_height = image_height
    self.image_width = image_width
    self.label_height = label_height
    self.label_width = label_width
    self.batch_size = batch_size
    self.dropout_keep_prob=0.5

    self.c_dim = c_dim
    self.df_dim = 64
    self.checkpoint_dir = checkpoint_dir
    self.c_depth_dim=c_depth_dim
    self.build_model()

  def build_model(self):
    self.images = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images')
    self.depth = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width,self.c_depth_dim], name='depth')
    self.pred_h = self.model()
    self.saver = tf.train.Saver()
     
  def train(self, image_test, depth_test, test_image_name, config):


    # Stochastic gradient descent with the standard backpropagation,var_list=self.model_c_vars

    shape = image_test.shape
    expand_test = image_test[np.newaxis,:,:,:]
    expand_zero = np.zeros([self.batch_size-1,shape[0],shape[1],shape[2]])
    batch_test_image = np.append(expand_test,expand_zero,axis = 0)


    shape = image_test.shape
    shape1 = depth_test.shape
  
    expand_test1 = depth_test[np.newaxis,:,:]
    expand_zero1 = np.zeros([self.batch_size-1,shape1[0],shape1[1]])
    batch_test_depth1 = np.append(expand_test1,expand_zero1,axis = 0)
    batch_test_depth= batch_test_depth1.reshape(self.batch_size,shape1[0],shape1[1],1)

    counter = 0

    start_time = time.time()
    result_h  = self.sess.run(self.pred_h, feed_dict={self.images: batch_test_image, self.depth: batch_test_depth})

    _,h ,w , c = result_h.shape
    for id in range(0,1):
        result_h01 = result_h[id,:,:,:].reshape(h , w , 1)
        result_h0 = result_h01.squeeze()

        image_path0 = os.path.join(os.getcwd(), config.sample_dir)
        image_path = os.path.join(image_path0, test_image_name+'_out.png')
        imsave_lable(result_h0, image_path)

  def model(self):
    with tf.variable_scope("depth_rgb_branch") as scope:

      conv1_c = tf.nn.relu(conv2d(self.images, 1,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_1"))
      conv1_f = tf.nn.relu(conv2d(self.depth, 1,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_1f"))

      conv1_rgb_c = tf.nn.relu(conv2d(conv1_f, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_rgb_c"))
      conb1_rgbd_c = tf.concat(axis = 3, values = [conv1_c,conv1_rgb_c]) #
      conv2_c = tf.nn.relu(conv2d(conb1_rgbd_c, 64,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_2"))
      pool1_c=max_pool_2x2(conv2_c)

      conv1_rgb = tf.nn.relu(conv2d(conv1_c, 64,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_rgb"))#
      conb1_rgbd = tf.concat(axis = 3, values = [conv1_f,conv1_rgb]) #
      conv2_f = tf.nn.relu(conv2d(conb1_rgbd, 64,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_2f"))
      pool1_f=max_pool_2x2(conv2_f)

      conv3_c = tf.nn.relu(conv2d(pool1_c, 128,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_3"))
      conv3_f = tf.nn.relu(conv2d(pool1_f, 128,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_3f"))
      conv3_rgb_c = tf.nn.relu(conv2d(conv3_f, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv3_rgb_c"))#
      conb3_rgbd_c = tf.concat(axis = 3, values = [conv3_c,conv3_rgb_c]) 
      conv4_c = tf.nn.relu(conv2d(conb3_rgbd_c, 128,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_4"))
      pool2_c=max_pool_2x2(conv4_c)

      conv3_rgb = tf.nn.relu(conv2d(conv3_c, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv3_rgb"))#
      conb3_rgbd = tf.concat(axis = 3, values = [conv3_f,conv3_rgb]) #
      conv4_f = tf.nn.relu(conv2d(conb3_rgbd, 128,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_4f"))
      pool2_f=max_pool_2x2(conv4_f)

      conv5_c = tf.nn.relu(conv2d(pool2_c, 256,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_5"))
      conv5_f = tf.nn.relu(conv2d(pool2_f, 256,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_5f"))
      conv5_rgb_c = tf.nn.relu(conv2d(conv5_f, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv5_rgb_c"))#
      conb5_rgbd_c = tf.concat(axis = 3, values = [conv5_c,conv5_rgb_c]) #
      conv6_c = tf.nn.relu(conv2d(conb5_rgbd_c, 256,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_6"))
      conv5_rgb = tf.nn.relu(conv2d(conv5_c, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv5_rgb"))#      
      conb5_rgbd = tf.concat(axis = 3, values = [conv5_f,conv5_rgb]) #
      conv6_f = tf.nn.relu(conv2d(conb5_rgbd, 256,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_6f"))
      conv6_rgb_c = tf.nn.relu(conv2d(conv6_f, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv6_rgb_c"))#
      conb6_rgbd_c = tf.concat(axis = 3, values = [conv6_c,conv6_rgb_c]) #
      conv7_c = tf.nn.relu(conv2d(conb6_rgbd_c, 256,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_7"))
      pool3_c=max_pool_2x2(conv7_c)

      conv6_rgb = tf.nn.relu(conv2d(conv6_c, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv6_rgb"))#
      conb6_rgbd = tf.concat(axis = 3, values = [conv6_f,conv6_rgb]) #
      conv7_f = tf.nn.relu(conv2d(conb6_rgbd, 256,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_7f"))
      pool3_f=max_pool_2x2(conv7_f)

      conv8_c = tf.nn.relu(conv2d(pool3_c, 256,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_8"))
      conv8_f = tf.nn.relu(conv2d(pool3_f, 256,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_8f"))
      conv8_rgb_c= tf.nn.relu(conv2d(conv8_f, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv8_rgb_c"))#
      conb8_rgbd_c = tf.concat(axis = 3, values = [conv8_c,conv8_rgb_c]) #
      conv9_c = tf.nn.relu(conv2d(conb8_rgbd_c, 256, 256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_9"))
      conv8_rgb = tf.nn.relu(conv2d(conv8_c, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv8_rgb"))#
      conb8_rgbd = tf.concat(axis = 3, values = [conv8_f,conv8_rgb]) #
      conv9_f = tf.nn.relu(conv2d(conb8_rgbd, 256, 256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_9f"))      
      conv9_rgb_c = tf.nn.relu(conv2d(conv9_f, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv9_rgb_c"))#
      conb9_rgbd_c = tf.concat(axis = 3, values = [conv9_c,conv9_rgb_c]) #
      conv10_c = tf.nn.relu(conv2d(conb9_rgbd_c, 256, 256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_10"))
      pool4_c=max_pool_2x2(conv10_c)

      conv9_rgb = tf.nn.relu(conv2d(conv9_c, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv9_rgb"))#
      conb9_rgbd = tf.concat(axis = 3, values = [conv9_f,conv9_rgb]) #
      conv10_f = tf.nn.relu(conv2d(conb9_rgbd, 256, 256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_10f"))
      pool4_f=max_pool_2x2(conv10_f)

      conv11_f = tf.nn.relu(conv2d(pool4_f, 512,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_11f"))
      conv11_c = tf.nn.relu(conv2d(pool4_c, 512,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_11"))
      conv11_rgb_c = tf.nn.relu(conv2d(conv11_f, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv11_rgb_c"))#
      conv11_rgb = tf.nn.relu(conv2d(conv11_c, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv11_rgb"))#
      conb11_rgbd = tf.concat(axis = 3, values = [conv11_f,conv11_rgb]) # 
      conb11_rgbd_c = tf.concat(axis = 3, values = [conv11_c,conv11_rgb_c]) # 
      conv12_c = tf.nn.relu(conv2d(conb11_rgbd_c, 512, 512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_12"))
      conv12_f = tf.nn.relu(conv2d(conb11_rgbd, 512, 512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_12f")) 
      conv12_rgb_c = tf.nn.relu(conv2d(conv12_f, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv12_rgb_c"))#
      conb12_rgbd_c = tf.concat(axis = 3, values = [conv12_c,conv12_rgb_c]) #
      conv13_c = tf.nn.relu(conv2d(conb12_rgbd_c, 512, 512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_13"))

      conv12_rgb = tf.nn.relu(conv2d(conv12_c, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv12_rgb"))#
      conb12_rgbd = tf.concat(axis = 3, values = [conv12_f,conv12_rgb]) #      
      conv13_f = tf.nn.relu(conv2d(conb12_rgbd, 512, 512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_13f"))


    with tf.variable_scope("fusion_branch") as scope:

      conb1 = tf.concat(axis = 3, values = [conv13_c,conv13_f])
      conv1_h = tf.nn.relu(conv2d(conb1, 1024,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_1"))
      conb2 = tf.concat(axis = 3, values = [conv12_c,conv12_f,conv1_h])
      conv2_h = tf.nn.relu(conv2d(conb2, 1024,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_2"))
      conb3 = tf.concat(axis = 3, values = [conv11_c,conv11_f,conv2_h])
      conv3_h = conv2d(conb3, 1024,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_3")
      conv3_dh3 =tf.nn.relu(deconv2d(conv3_h, conv10_c.get_shape().as_list(), k_h=3, k_w=3, d_h=2, d_w=2,name="deconv2d_1_3"))
      conb4 = tf.concat(axis = 3, values = [conv10_c,conv10_f,conv3_dh3])

      conv4_h = tf.nn.relu(conv2d(conb4, 1024,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_4"))
      side_out1 = conv2d(conb4, 1024,1,k_h=3, k_w=3, d_h=1, d_w=1,name="side_out1")
      conv9_c_enhanced=conv9_c+tf.multiply(tf.nn.sigmoid(side_out1),conv9_c)
      conv9_f_enhanced=conv9_f+tf.multiply(tf.nn.sigmoid(side_out1),conv9_f)
      conb5 = tf.concat(axis = 3, values = [conv9_c_enhanced,conv9_f_enhanced,conv4_h])
      conv5_h = tf.nn.relu(conv2d(conb5, 1024,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_5"))
      conv8_c_enhanced=conv8_c+tf.multiply(tf.nn.sigmoid(side_out1),conv8_c)
      conv8_f_enhanced=conv8_f+tf.multiply(tf.nn.sigmoid(side_out1),conv8_f)
      conb6 = tf.concat(axis = 3, values = [conv8_c_enhanced,conv8_f_enhanced,conv5_h])
      conv6_h = tf.nn.relu(conv2d(conb6, 1024,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_6"))
      conv6_dh3 =tf.nn.relu(deconv2d(conv6_h, conv7_c.get_shape().as_list(), k_h=3, k_w=3, d_h=2, d_w=2,name="deconv2d_2_3"))

      conb7 = tf.concat(axis = 3, values = [conv7_c,conv7_f,conv6_dh3])
      conv7_h = tf.nn.relu(conv2d(conb7, 1024,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_7"))
      side_out2 = conv2d(conb7, 1024,1,k_h=3, k_w=3, d_h=1, d_w=1,name="side_out2")
      conv6_c_enhanced=conv6_c+tf.multiply(tf.nn.sigmoid(side_out2),conv6_c)
      conv6_f_enhanced=conv6_f+tf.multiply(tf.nn.sigmoid(side_out2),conv6_f)
      conb8 = tf.concat(axis = 3, values = [conv6_c_enhanced,conv6_f_enhanced,conv7_h])
      conv8_h = tf.nn.relu(conv2d(conb8, 1024,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_8"))
      conv5_c_enhanced=conv5_c+tf.multiply(tf.nn.sigmoid(side_out2),conv5_c)
      conv5_f_enhanced=conv5_f+tf.multiply(tf.nn.sigmoid(side_out2),conv5_f)
      conb9 = tf.concat(axis = 3, values = [conv5_c_enhanced,conv5_f_enhanced,conv8_h])
      conv9_h = tf.nn.relu(conv2d(conb9, 1024,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_9"))
      conv9_dh3 =tf.nn.relu(deconv2d(conv9_h, conv4_c.get_shape().as_list(), k_h=3, k_w=3, d_h=2, d_w=2,name="deconv2d_3_3"))      
      conb10 = tf.concat(axis = 3, values = [conv4_c,conv4_f,conv9_dh3])
      conv10_h = tf.nn.relu(conv2d(conb10, 1024,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_10"))
      side_out3 =conv2d(conb10, 1024,1,k_h=3, k_w=3, d_h=1, d_w=1,name="side_out3")
      conv3_c_enhanced=conv3_c+tf.multiply(tf.nn.sigmoid(side_out3),conv3_c)
      conv3_f_enhanced=conv3_f+tf.multiply(tf.nn.sigmoid(side_out3),conv3_f)  
      conb11 = tf.concat(axis = 3, values = [conv3_c_enhanced,conv3_f_enhanced,conv10_h])
      conv11_h = tf.nn.relu(conv2d(conb11, 1024,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_11"))
      conv11_dh3 =tf.nn.relu(deconv2d(conv11_h, conv2_c.get_shape().as_list(), k_h=3, k_w=3, d_h=2, d_w=2,name="deconv2d_4_3"))  
      conb12 = tf.concat(axis = 3, values = [conv2_c,conv2_f,conv11_dh3])
      conv12_h = tf.nn.relu(conv2d(conb12, 1024,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_12"))     
      conb13 = tf.concat(axis = 3, values = [conv1_c,conv1_f,conv12_h])
      conv13_h = conv2d(conb13, 1024,1,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_13")
    return tf.nn.sigmoid(conv13_h)
   


  def save(self, checkpoint_dir, step):
    model_name = "coarse.model"
    model_dir = "%s_%s" % ("coarse", self.label_height)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
 
    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("coarse", self.label_height)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    # import pdb
    # pdb.set_trace()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False