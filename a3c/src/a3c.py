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
from autoencoder import autoencoder_network
import tensorflow as tf
from keras import backend as K

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU


class actor_critic:

  def __init__(self):     
    random.seed()
    
    self.colliding_flag = False
    
    # Set to 1 to activate training and to 0 to deactivate (reading from a ros parameter)
    self.train_indicator = rospy.get_param('~training')
    
    # Reading the networks' path from a ros parameter
    self.networks_dir = rospy.get_param('~networks_dir')
    
    # Reading the input modality from a ros parameter (laser or camera)
    self.input_mode = rospy.get_param('~input')
    
    # Set to 1 to activate imu inputs and to 0 to deactivate (reading from a ros parameter)
    self.imu_input_mod = rospy.get_param('~imu_input')
    
    self.queue = deque([])
    
    self.graph = tf.get_default_graph()    
    
    if self.imu_input_mod == 1:
      self.imu_dim = 37
    else:
      self.imu_dim = 0
    if self.input_mode == "laser":
      self.feature_dim = 9
      self.aruco_dim = 3
      self.altitude_dim = 1
    else:
      self.feature_dim = 200
      self.aruco_dim = 0
      self.altitude_dim = 0
    self.state_dim = (3 * self.feature_dim) + self.imu_dim + self.aruco_dim + self.altitude_dim
    self.action_dim = 3
    self.buffer_size = 100000
    self.batch_size = 32
    self.gamma = 0.99
    self.tau = 0.001     #Target Network HyperParameters
    self.lra = 0.0001    #Learning rate for Actor
    self.lrc = 0.001     #Lerning rate for Critic
    self.epsilon = 1
    
    self.ou = OU()
    
    # Initialization of state, action and noise vectors
    self.action = np.zeros((1, self.action_dim))
    self.action_noise = np.zeros((1, self.action_dim))
    self.state = np.zeros(self.state_dim)
    
    with self.graph.as_default():
      # Initialization of the autoencoder network
      self.autoencoder_network = autoencoder_network(self.networks_dir)
      #Tensorflow GPU optimization
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      sess = tf.Session(config=config)
      K.set_session(sess)
      # Initialization of the actor and critic networks
      self.actor = ActorNetwork(sess, self.state_dim, self.action_dim, self.batch_size, self.tau, self.lra)
      self.critic = CriticNetwork(sess, self.state_dim, self.action_dim, self.batch_size, self.tau, self.lrc)    
      #Now load the weight
      rospy.loginfo("Now we load the weight")
      try:
          self.actor.model.load_weights(self.networks_dir + "actormodel.h5")
          self.critic.model.load_weights(self.networks_dir + "criticmodel.h5")
          self.actor.target_model.load_weights(self.networks_dir + "actormodel.h5")
          self.critic.target_model.load_weights(self.networks_dir + "criticmodel.h5")
          rospy.loginfo("Weight load successfully")
      except:
          rospy.logwarn("Cannot find the weight")
    
    #Create replay buffer
    self.buff = ReplayBuffer(self.buffer_size)
      
    self.aruco_limit = 4.0
    self.altitude_limit = 2.0
    self.imu_msg = np.zeros(self.imu_dim)
    self.bumper_msg = None
    self.navdata_msg = None
    self.aruco_msg = None
    self.laser_msg = None
    self.model_states_pose_msg = None    
    self.count = 0
    self.step = 0
    self.bridge = CvBridge()

    actual_time = rospy.get_rostime()
    self.start_time =  actual_time.secs + \
                       actual_time.nsecs / 1000000000
    
    self.start_pos = Twist()
    self.start_pos.linear.x = 0.0
    self.start_pos.linear.y = 0.0
    self.start_pos.linear.z = 0.0
    
    self.total_reward = 0.0
    
    self.position_list = [[0, 0], [20, 0], [20, -20], [0, -20], [0, -7], [7, -7]]
    
    # Subscribers initialization
    self.image_sub = rospy.Subscriber("/ardrone/front/image_raw",
                                      Image, self.callback_image)
    self.imu_sub = rospy.Subscriber("/ardrone/imu",
                                    Imu, self.callback_imu)
    self.bumper_sub = rospy.Subscriber("/ardrone/bumper",
                                       ContactsState, self.callback_bumper)
    self.navdata_sub = rospy.Subscriber("/ardrone/navdata",
                                       Navdata, self.callback_navdata)
    self.aruco_sub = rospy.Subscriber("/ardrone/aruco/pose",
                                       Twist, self.callback_aruco)
    self.model_states_sub = rospy.Subscriber("/gazebo/model_states",
                                             ModelStates, self.callback_model_states)
    self.laser_sub = rospy.Subscriber("/ardrone/laser",
                                      LaserScan, self.callback_laser)
                                      
    rospy.loginfo("Subscribers initialized")
    
    # Publishers initialization
    self.model_state_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
    self.reset_pub = rospy.Publisher('/ardrone/reset', Empty, queue_size=10)
    self.takeoff_pub = rospy.Publisher('/ardrone/takeoff', Empty, queue_size=10)
    self.land_pub = rospy.Publisher('/ardrone/land', Empty, queue_size=10)
    self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    
    rospy.loginfo("Publishers initialized")
    
    # Small sleep to give time to publishers to open the topics
    rospy.sleep(0.5)     
    
    rospy.loginfo("taking off")
    self.takeoff_pub.publish(Empty())

  def callback_image(self,data):
    if (self.navdata_msg != None and \
        self.navdata_msg.state == 3):  # The actor-critic system works only when the drone is flying
        
        rospy.logdebug("Episode : " + str(self.count) + " Replay Buffer " + str(self.buff.count())) 
        
        loss = 0

        # Rmoving inf from laser ranges
        if self.laser_msg != None:
          laser_ranges = np.asarray(self.laser_msg.ranges)
          laser_ranges[laser_ranges == inf] = 0
        
        # Calculating laser punishment
        laser_cost = 0.0
        if self.laser_msg != None:
          inverted_ranges = 1 - (laser_ranges / self.laser_msg.range_max)
          gaussian_ranges = np.multiply(inverted_ranges, signal.gaussian(self.feature_dim, (self.feature_dim / 2 * 0.8)))
          laser_cost = -np.sum(gaussian_ranges) / self.feature_dim
        rospy.loginfo("Laser range punishment: " + str(laser_cost))
        
        # Calculating the punishment for colliding
        is_colliding = False
        collision_cost = 0.0
        if self.colliding_flag:
            is_colliding = True
            collision_cost = -10.0
        rospy.logdebug("Collisions punishment: " + str(collision_cost))
            
        # Calculating the time elapsed from the last respawn
        actual_time = rospy.get_rostime()        
        time_stamp = actual_time.secs + \
                     actual_time.nsecs / 1000000000                    
        time_elapsed = time_stamp - self.start_time
        rospy.logdebug("Time elapsed: " + str(time_elapsed))
        
        # Calculating the aruco distance reward
        aruco_dist = 0.0
        aruco_cost = 0.0        
        if self.aruco_msg != None:
            aruco_dist = np.sqrt(self.aruco_msg.linear.x**2 + \
                                 self.aruco_msg.linear.y**2 + \
                                 self.aruco_msg.linear.z**2)
            if aruco_dist == 0.0 or aruco_dist > self.aruco_limit:
                aruco_dist = self.aruco_limit
            aruco_cost = 1.0 - (aruco_dist / self.aruco_limit)
            rospy.logdebug("Aruco distance reward: " + str(aruco_cost))
            
        # Calculating the traveled distance reward and the altitude punishment
        trav_dist = 0.0
        alt_cost = 0.0
        if self.model_states_pose_msg != None:
          actual_pos = self.model_states_pose_msg.position
          trav_dist = np.sqrt((actual_pos.x - self.start_pos.linear.x)**2 + \
                              (actual_pos.y - self.start_pos.linear.y)**2)
          trav_cost = trav_dist / time_elapsed
          alt_cost = -abs(1 - actual_pos.z)
        rospy.logdebug("Travel distance reward: " + str(trav_cost))
    
        # Calculating the angular velocity punishment
        angular_cost = 0
        if self.imu_msg != None:
            angular_cost = -abs(self.imu_msg[6])
        rospy.loginfo("Angular punishment: " + str(angular_cost))
        
        # Calculating the step reward
        step_reward = collision_cost + \
                      (10 * aruco_cost) + \
                      trav_cost + alt_cost \
                      + laser_cost #+ angular_cost
        rospy.logdebug("Step reward: " + str(step_reward))
        
        # Calculating the total reward
        self.total_reward += step_reward
        rospy.logdebug("Total reward: " + str(self.total_reward))
        
        image_features = np.zeros(self.feature_dim)
        if self.input_mode == "camera":
          rospy.logwarn(self.input_mode + " CAMERA")
          # Reading the camera image from the topic
          try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
          except CvBridgeError as err:
            rospy.logerror(err)
          
          # Risizeing the image to 160x80 pixel for the convolutional network
          cv_image_resized = cv2.resize(cv_image, (160, 80))
          #array_image = np.asarray(cv_image_resized)
          array_image = img_to_array(cv_image_resized)
          input_image = np.zeros((1, 80, 160, 3))
          input_image[0] = array_image
          input_image /= 255.0
          
          # Calculating image features
          with self.graph.as_default():
            image_features = self.autoencoder_network.run_network(input_image)
        elif self.laser_msg != None:
          rospy.logwarn(self.input_mode + " LASER")
          # Reading laser data as state features
          image_features = laser_ranges / self.laser_msg.range_max 
          
        # Adding the features to the features list
        if len(self.queue) == 0:
            self.queue.append(image_features)
            self.queue.append(image_features)
            self.queue.append(image_features)
        else:
            self.queue.popleft()
            self.queue.append(image_features)
            
        rospy.logdebug("Queue length: " + str(len(self.queue)))
        rospy.logdebug("Image Features: " + str(image_features.shape))
        rospy.logdebug("Imu length: " + str(len(self.imu_msg)))
        
        # Create the state vector
        new_state = np.concatenate((self.queue[0].flatten(), 
                                    self.queue[1].flatten(),
                                    self.queue[2].flatten()))
        if self.imu_input_mod == 1:
          new_state = np.concatenate((new_state,
                                      self.imu_msg))
        if self.input_mode == "laser":
          # Creating aruco features
          aruco_features = np.zeros(self.aruco_dim)
          if self.aruco_msg != None:
              if self.aruco_msg.linear.x == 0 or self.aruco_msg.linear.x > self.aruco_limit:
                  aruco_features[0] = 1.0
              else:
                  aruco_features[0] = self.aruco_msg.linear.x / self.aruco_limit
              if self.aruco_msg.linear.y == 0 or self.aruco_msg.linear.y > self.aruco_limit:
                  aruco_features[1] = 1.0
              else:
                  aruco_features[1] = self.aruco_msg.linear.y / self.aruco_limit
              if self.aruco_msg.linear.z == 0 or self.aruco_msg.linear.y > self.aruco_limit:
                  aruco_features[2] = 1.0
              else:
                  aruco_features[2] = self.aruco_msg.linear.z / self.aruco_limit
              #aruco_features[3] = self.aruco_msg.angular.x / np.pi
              #aruco_features[4] = self.aruco_msg.angular.y / np.pi
              #aruco_features[5] = self.aruco_msg.angular.z / np.pi
            
          # Creating altitude feature         
          altitude_feature = np.zeros(1)        
          if self.model_states_pose_msg != None:
            altitude_value = self.model_states_pose_msg.position.z
            if altitude_value > self.altitude_limit:
                altitude_feature[0] = 1.0
            else:
                altitude_feature[0] = altitude_value / self.altitude_limit
          new_state = new_state = np.concatenate((new_state,
                                                  aruco_features,
                                                  altitude_feature))
        
        rospy.logdebug("State length: " + str(len(new_state)))
        rospy.loginfo("State: " + str(new_state))
        
        # Add replay buffer
        done = False
        if is_colliding or \
           (aruco_cost > 0.8):
               done = True
        self.buff.add(self.state, self.action[0], step_reward, new_state, done)
        
        # Calculating new action
        with self.graph.as_default():
          a_t_original = self.actor.model.predict(new_state.reshape(1, new_state.shape[0]))
        self.action_noise[0][0] = self.train_indicator * max(self.epsilon, 0) * \
                               self.ou.function(a_t_original[0][0],  0.3, 0.5, 0.1)
        self.action_noise[0][1] = self.train_indicator * max(self.epsilon, 0) * \
                               self.ou.function(a_t_original[0][1],  0.0, 0.5, 0.1)
        self.action_noise[0][2] = self.train_indicator * max(self.epsilon, 0) * \
                               self.ou.function(a_t_original[0][2], 0.0, 0.5, 0.1)

        self.action[0][0] = a_t_original[0][0] + self.action_noise[0][0]
        self.action[0][1] = a_t_original[0][1] + self.action_noise[0][1]
        self.action[0][2] = a_t_original[0][2] + self.action_noise[0][2]
#        with self.graph.as_default():
#          a_t_original = self.actor.model.predict(new_state.reshape(1, new_state.shape[0]))
#        self.action_noise[0][0] = self.train_indicator * max(self.epsilon, 0) * \
#                               self.ou.function(a_t_original[0][0],  0.3, 0.6, 0.1)
#        self.action_noise[0][1] = self.train_indicator * max(self.epsilon, 0) * \
#                               self.ou.function(a_t_original[0][1],  0.0, 0.5, 0.1)
#        self.action_noise[0][2] = self.train_indicator * max(self.epsilon, 0) * \
#                               self.ou.function(a_t_original[0][2], 0.0, 0.9, 0.1)
#        self.action_noise[0][3] = self.train_indicator * max(self.epsilon, 0) * \
#                               self.ou.function(a_t_original[0][3], 0.0, 0.5, 0.1)
#
#        self.action[0][0] = a_t_original[0][0] + self.action_noise[0][0]
#        self.action[0][1] = a_t_original[0][1] + self.action_noise[0][1]
#        self.action[0][2] = a_t_original[0][2] + self.action_noise[0][2]
#        self.action[0][3] = a_t_original[0][3] + self.action_noise[0][3]
        
        rospy.loginfo("motor comand plus noise: " + str( self.action[0][2]) + \
                      " original motor command: " + str(a_t_original[0][2]) + \
                      " noise: " + str(self.action_noise[0][2]))

        # Perform an action
        cmd_input = Twist()
#        cmd_input.linear.x = self.action[0][0]
#        cmd_input.linear.y = self.action[0][1]
#        cmd_input.linear.z = self.action[0][2]
#        cmd_input.angular.z = self.action[0][3]
        cmd_input.linear.x = self.action[0][0]
        cmd_input.linear.y = 0.0
        cmd_input.linear.z = self.action[0][1]
        # CAMBIO TEST MOMENTANEOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#        if abs(a_t_original[0][2]) > 0.7:
#            self.action[0][2] = 0.0
        cmd_input.angular.z = self.action[0][2]
        self.cmd_vel_pub.publish(cmd_input)
        
        # Updating the state
        self.state = new_state
        
        #Do the batch update
        batch = self.buff.getBatch(self.batch_size)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        y_t = np.asarray([e[1] for e in batch])

        # Calculating Q value 
        with self.graph.as_default():
          target_q_values = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])  
       
        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + self.gamma * target_q_values[k]
        
        # Training the network
        with self.graph.as_default():
            if (self.train_indicator):
                loss += self.critic.model.train_on_batch([states, actions], y_t) 
                a_for_grad = self.actor.model.predict(states)
                grads = self.critic.gradients(states, a_for_grad)
                self.actor.train(states, grads)
                self.actor.target_train()
                self.critic.target_train()
                
        rospy.logdebug("Episode: " + str(self.count) + \
                      " Step: " + str(self.step) + \
                      " Action: " + str(self.action) + \
                      " Reward: " + str(step_reward) + \
                      " Loss: " + str(loss))
                
        self.step +=1
        
        # Starting a newe episode if the drone collide with something or if it's close enough to an aruco board
        if is_colliding or \
           (aruco_cost > 0.8):
            model_state_msg = ModelState()
            empty_msg = Empty()
            
            # Generating a new random position chosen between 4
            new_position = random.sample(self.position_list,  1)
            
            # Generating a random orientation
            angle = random.random() * 2 * np.pi
            
            rospy.loginfo("New position: " + str(new_position) + \
                          "New angle: " + str(angle))
            
            # Creating the model state message to send to set_model_space topic
            model_state_msg.model_name = "quadrotor"
            model_state_msg.pose.position.x = new_position[0][0]
            model_state_msg.pose.position.y = new_position[0][1]
            model_state_msg.pose.position.z = 0.0
            quaternion = quaternion_from_euler(0, 0, angle)
            model_state_msg.pose.orientation.x = quaternion[0]
            model_state_msg.pose.orientation.y = quaternion[1]
            model_state_msg.pose.orientation.z = quaternion[2]
            model_state_msg.pose.orientation.w = quaternion[3]
            model_state_msg.reference_frame = "world"
            
            self.start_pos.linear.x = new_position[0][0]
            self.start_pos.linear.y = new_position[0][1]
            self.start_pos.linear.z = 0.0
            
            # Reseting the episode starting time
            actual_time = rospy.get_rostime()
            self.start_time =  actual_time.secs + \
                               actual_time.nsecs / 1000000000
            # Reseting the image queue
            self.queue = deque([])
            # Reseting state, action and noise vectors
            self.action = np.zeros((1, self.action_dim))
            self.action_noise = np.zeros((1, self.action_dim))
            self.state = np.zeros(self.state_dim)
            
            # Saving the weights
            with self.graph.as_default():
                if (self.train_indicator):
                    rospy.loginfo("Saving the weights")
                    self.actor.model.save_weights(self.networks_dir + "actormodel.h5", overwrite=True)
                    with open(self.networks_dir + "actormodel.json", "w") as outfile:
                        json.dump(self.actor.model.to_json(), outfile)
    
                    self.critic.model.save_weights(self.networks_dir + "criticmodel.h5", overwrite=True)
                    with open(self.networks_dir + "criticmodel.json", "w") as outfile:
                        json.dump(self.critic.model.to_json(), outfile)

            rospy.loginfo("TOTAL REWARD @ " + str(self.count) +"-th Episode  : Reward " + str(self.total_reward))
            rospy.loginfo("Total Step: " + str(self.step))
            self.total_reward = 0.0
            self.count += 1
            self.step = 0
    
             # reset the actions
            cmd_input.linear.x = 0.0
            cmd_input.linear.y = 0.0
            cmd_input.linear.z = 0.0
            cmd_input.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd_input)
         
            # Reinitializing position, orientation and status of the drone
            self.land_pub.publish(empty_msg)
            self.model_state_pub.publish(model_state_msg)
            self.reset_pub.publish(empty_msg)
            rospy.sleep(0.5)
            self.takeoff_pub.publish(empty_msg)
            rospy.sleep(0.5)
            self.colliding_flag = False
            

  def callback_imu(self,data):
    time_stamp = data.header.stamp.secs + \
                 data.header.stamp.nsecs / 1000000000
                 
    orientation = np.zeros(4)
    orientation[0] = data.orientation.x
    orientation[1] = data.orientation.y
    orientation[2] = data.orientation.z
    orientation[3] = data.orientation.w
    
    angular_velocity = np.zeros(3)
    angular_velocity[0] = data.angular_velocity.x
    angular_velocity[1] = data.angular_velocity.y
    angular_velocity[2] = data.angular_velocity.z
    
    linear_acceleration = np.zeros(3)
    linear_acceleration[0] = data.linear_acceleration.x / 10
    linear_acceleration[1] = data.linear_acceleration.y / 10
    # Normalizing the compensation for the gravity in z
    linear_acceleration[2] = data.linear_acceleration.z / 10 
    
    self.imu_msg = np.concatenate((orientation, 
                                   angular_velocity,
                                   linear_acceleration,
                                   data.orientation_covariance,
                                   data.angular_velocity_covariance,
                                   data.linear_acceleration_covariance))
    
  def callback_bumper(self,data):                 
    self.bumper_msg = data
    if self.bumper_msg.states != []:
        self.colliding_flag = True
    
  def callback_navdata(self,data): 
    # 0: Unknown, 1: Init, 2: Landed, 3: Flying, 4: Hovering, 5: Test
    # 6: Taking off, 7: Goto Fix Point, 8: Landing, 9: Looping
    # Note: 3,7 seems to discriminate type of flying (isFly = 3 | 7)                 
    self.navdata_msg = data
    
  def callback_aruco(self,data):                 
    self.aruco_msg = data
    
  def callback_laser(self,data):                 
    self.laser_msg = data
    
  def callback_model_states(self,data):
    is_there = -1
    for i in range(len(data.name)):
      if data.name[i] == "quadrotor":
        is_there = i

    if is_there >= 0:
      self.model_states_pose_msg = data.pose[is_there]
    else:
      self.model_states_pose_msg = None         
        
def main(args):
  #cv2.startWindowThread()
  #cv2.namedWindow("Image window")
  rospy.init_node('a3c', anonymous=True, log_level=rospy.INFO)
  
  #ac = actor_critic()
  time.sleep(1)
  
  rospy.loginfo("<------Data recorder (Author: Nino Cauli)------>")
  
  # Generating a random orientation
  random.seed()
  world_step = 30
  position_list = [[0, 0], [20, 0], [20, -22], [0, -20], [0, -7], 
                   [7, -7], [20, -17], [12, -10], [20, -13], [16, -17], 
                   [16, -3], [2, -3], [2, -17]]
  map_pos = [9, -10]
  aruco_pos_list = [[-4.5, 0, 1.0, 0, 1.57, 0], 
                    [8.60, -10.0, 1.0, 0, 1.57, 0],
                    [23.0, 0, 1.0, 0, 1.57, 3.14],
                    [-4.5, -20, 1.0, 0, 1.57, 0]]
  robot_description = rospy.get_param('~robot_description')
  map_path = rospy.get_param('~world_name')
  aruco_path = rospy.get_param('~aruco_name')
  f = open(map_path,'r')
  map_description = f.read()
  f.close()
  f = open(aruco_path,'r')
  aruco_description = f.read()
  f.close()
  rospy.wait_for_service('gazebo/spawn_urdf_model')
  rospy.wait_for_service('gazebo/spawn_sdf_model')
  spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_urdf_model', SpawnModel)
  spawn_model_sdf_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
  
  num_drone = 0
  for start_pos in position_list:  
      num_drone += 1
      
      quadrotor_name = "quadrotor_" + str(num_drone)
      name_space = "number_" + str(num_drone)
      map_name = "map_" + str(num_drone)
      
      initial_pose = Pose()
      initial_pose.position.x = map_pos[0]
      initial_pose.position.y = map_pos[1] + (world_step * (num_drone - 1))
      initial_pose.position.z = 0.0
      
      spawn_model_sdf_prox(map_name, map_description, name_space, initial_pose, "world")    
      
      num_aruco_board = 0      
      for aruco_boards_pos in aruco_pos_list:
          num_aruco_board += 1
          
          aruco_name = "aruco_" + str(num_drone) + "_" + str(num_aruco_board)
          initial_pose = Pose()
          initial_pose.position.x = aruco_boards_pos[0]
          initial_pose.position.y = aruco_boards_pos[1] + (world_step * (num_drone - 1))
          initial_pose.position.z = aruco_boards_pos[2]
          quaternion = quaternion_from_euler(aruco_boards_pos[3], 
                                             aruco_boards_pos[4], 
                                             aruco_boards_pos[5])
          initial_pose.orientation.x = quaternion[0]
          initial_pose.orientation.y = quaternion[1]
          initial_pose.orientation.z = quaternion[2]
          initial_pose.orientation.w = quaternion[3]
          
          spawn_model_sdf_prox(aruco_name, aruco_description, name_space, initial_pose, "world")
      
      initial_pose = Pose()
      initial_pose.position.x = start_pos[0]
      initial_pose.position.y = start_pos[1] + (world_step * (num_drone - 1))
      initial_pose.position.z = 0.5
      angle = random.random() * 2 * np.pi
      quaternion = quaternion_from_euler(0, 0, angle)
      initial_pose.orientation.x = quaternion[0]
      initial_pose.orientation.y = quaternion[1]
      initial_pose.orientation.z = quaternion[2]
      initial_pose.orientation.w = quaternion[3]
  
      spawn_model_prox(quadrotor_name, robot_description, name_space, initial_pose, "world")
  
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")   
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
