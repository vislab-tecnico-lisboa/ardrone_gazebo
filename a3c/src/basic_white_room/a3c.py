#!/usr/bin/env python
from __future__ import print_function, division

import roslib
import random
roslib.load_manifest('a3c')
import sys
import rospy
import numpy as np
from scipy import signal
import cv2
from sensor_msgs.msg import Image, Imu, LaserScan
from geometry_msgs.msg import Twist, Pose
from gazebo_msgs.msg import ModelState, ModelStates, ContactsState
from gazebo_msgs.srv import SpawnModel
from ardrone_autonomy.msg import Navdata
from std_msgs.msg import Empty
from cv_bridge import CvBridge, CvBridgeError
import time
import os
from collections import deque
from tf.transformations import quaternion_from_euler
import json
from numpy import inf

from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
from keras import backend as K

from worker import Worker
from ac_network import AC_Network

import threading
import multiprocessing      
        
def main(args):
  #cv2.startWindowThread()
  #cv2.namedWindow("Image window")
  rospy.init_node('a3c', anonymous=True, log_level=rospy.INFO)
  
  #ac = actor_critic()
  time.sleep(1)
  
  rospy.loginfo("<------A3C (Author: Nino Cauli)------>")
  
  net_path = rospy.get_param('~networks_dir')  
  
  max_episode_length = 3000
  gamma = .99 # discount rate for advantage estimation and reward discounting
  s_size = 21168 # Observations are greyscale frames of 84 * 84 * 1
  a_size = 2 # Agent can move Left, Right, or Fire
  load_model = False
  model_path = net_path + "/model"
  # TO BE CHANGED
  test = 0

  tf.reset_default_graph()

  if not os.path.exists(model_path):
      os.makedirs(model_path)
    
  #Create a directory to save episode playback gifs to
  if not os.path.exists(net_path + "/frames"):
      os.makedirs(net_path + "/frames")

  with tf.device("/cpu:0"): 
      global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
      trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
      master_network = AC_Network(s_size,a_size,'global',None) # Generate global network
      num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
      #num_workers = 4
      workers = []
      # Create worker classes
      for i in range(num_workers):
          print ("NUMBEEERR: " + str(i))
          workers.append(Worker(test,i,s_size,a_size,trainer,model_path,global_episodes,net_path))
      saver = tf.train.Saver(max_to_keep=5)

  with tf.Session() as sess:
      coord = tf.train.Coordinator()
      if load_model == True:
          print ('Loading Model...')
          ckpt = tf.train.get_checkpoint_state(model_path)
          saver.restore(sess,ckpt.model_checkpoint_path)
      else:
          sess.run(tf.global_variables_initializer())
        
      # This is where the asynchronous magic happens.
      # Start the "work" process for each worker in a separate threat.
      worker_threads = []
      for worker in workers:
          worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
          t = threading.Thread(target=(worker_work))
          t.start()
          rospy.sleep(0.5) 
          worker_threads.append(t)
      coord.join(worker_threads)  
  
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")   
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
