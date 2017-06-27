# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 16:40:08 2017

@author: Nino Cauli
"""

import math
import numpy as np
import tensorflow as tf
import scipy.signal
from helper import *

import roslib
import random
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

from ac_network import AC_Network


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Processes Doom screen image to produce cropped and resized image. 
def process_frame(frame):
    #rospy.loginfo("shape before: " + str(np.shape(frame)))
    #s = frame[10:-10,30:-30]
    s = frame
    rospy.logdebug("shape after cutting: " + str(np.shape(s)))
    s = scipy.misc.imresize(s,[84,84])
    s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    rospy.logdebug("shape after reshaping: " + str(np.shape(s)))
    return s

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
                
class Worker():
    def __init__(self,game,name,s_size,a_size,trainer,model_path,global_episodes,net_path):                     
        self.start_pos = Twist()
        self.start_pos.linear.x = 0.0
        self.start_pos.linear.y = 0.0
        self.start_pos.linear.z = 0.5
        
        self.name = "worker_" + str(name)
        print "NAMEEEEEEE: " + self.name
        self.number = name        
        self.model_path = model_path
        self.net_path = net_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        
        self.a_size = a_size

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size,a_size,self.name,trainer)
        self.update_local_ops = update_target_graph('global',self.name) 
        
        # Set to 1 to activate training and to 0 to deactivate (reading from a ros parameter)
        self.train_indicator = rospy.get_param('~training')
    
        # Reading the networks' path from a ros parameter
        self.networks_dir = rospy.get_param('~networks_dir')
        
        # Reading the input modality from a ros parameter (laser or camera)
        self.input_mode = rospy.get_param('~input')
        
        # Set to 1 to activate imu inputs and to 0 to deactivate (reading from a ros parameter)
        self.imu_input_mod = rospy.get_param('~imu_input')
        
        self.summary_writer = tf.summary.FileWriter(self.networks_dir + "/train_" + str(self.number))
        
        if self.imu_input_mod == 1:
          self.imu_dim = 37
        else:
          self.imu_dim = 0
        if self.input_mode == "laser":
          self.feature_dim = 9
          self.aruco_dim = 3
          self.altitude_dim = 1
        else:
          self.feature_dim = 9#200
          self.aruco_dim = 0
          self.altitude_dim = 0
        self.state_dim = (3 * self.feature_dim) + self.imu_dim + self.aruco_dim + self.altitude_dim
        self.action_dim = 3
        
        # Subscribers messages initialization
        self.imu_msg = np.zeros(self.imu_dim)
        self.bumper_msg = None
        self.navdata_msg = None
        self.aruco_msg = None
        self.laser_msg = None
        self.model_states_pose_msg = None
        self.img_msg = None
        self.colliding_flag = False
        
        # Flag to know if a new image arrived
        self.img_flag = False
        
        self.aruco_limit = 4.0
        self.altitude_limit = 2.0
        
        #ROS part
        # Generating a random orientation
        random.seed()
        self.world_step = 30
        self.position_list = [[0, 0], [20, 0], [20, -22], [0, -20], [0, -7], 
                         [7, -7], [20, -17], [12, -10], [20, -13], [16, -17], 
                       [16, -3], [2, -3], [2, -17]]
        map_pos = [9, -10]
        aruco_pos_list = [[-4.5, 0, 1.0, 0, 1.57, 0], 
                          [8.60, -10.0, 1.0, 0, 1.57, 0],
                          [23.0, 0, 1.0, 0, 1.57, 3.14],
                          [-4.5, -20, 1.0, 0, 1.57, 0]]
        random_index = int(random.random() * len(self.position_list))
        start_pos = self.position_list[random_index]
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
  
        self.num_drone = name
      
        self.quadrotor_name = "quadrotor_" + str(self.num_drone)
        name_space = "quadrotor_" + str(self.num_drone)
        map_name = "map_" + str(self.num_drone)
      
        initial_pose = Pose()
        initial_pose.position.x = map_pos[0]
        initial_pose.position.y = map_pos[1] + (self.world_step * (self.num_drone - 1))
        initial_pose.position.z = 0.0
      
        spawn_model_sdf_prox(map_name, map_description, name_space, initial_pose, "world")    
      
        num_aruco_board = 0      
        for aruco_boards_pos in aruco_pos_list:
            num_aruco_board += 1
          
            aruco_name = "aruco_" + str(self.num_drone) + "_" + str(num_aruco_board)
            initial_pose = Pose()
            initial_pose.position.x = aruco_boards_pos[0]
            initial_pose.position.y = aruco_boards_pos[1] + (self.world_step * (self.num_drone - 1))
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
        initial_pose.position.y = start_pos[1] + (self.world_step * (self.num_drone - 1))
        initial_pose.position.z = 0.5
        angle = random.random() * 2 * np.pi
        quaternion = quaternion_from_euler(0, 0, angle)
        initial_pose.orientation.x = quaternion[0]
        initial_pose.orientation.y = quaternion[1]
        initial_pose.orientation.z = quaternion[2]
        initial_pose.orientation.w = quaternion[3]
  
        spawn_model_prox(self.quadrotor_name, robot_description, name_space, initial_pose, "world")
        
         # Subscribers initialization
        self.image_sub = rospy.Subscriber("/" + name_space + "/ardrone/front/ardrone/front/image_raw",
                                          Image, self.callback_image)
        self.imu_sub = rospy.Subscriber("/" + name_space + "/ardrone/imu",
                                        Imu, self.callback_imu)
        self.bumper_sub = rospy.Subscriber("/" + name_space + "/ardrone/bumper",
                                           ContactsState, self.callback_bumper)
        self.navdata_sub = rospy.Subscriber("/" + name_space + "/ardrone/navdata",
                                            Navdata, self.callback_navdata)
        self.aruco_sub = rospy.Subscriber("/" + name_space + "/ardrone/aruco/pose",
                                            Twist, self.callback_aruco)
        self.model_states_sub = rospy.Subscriber("/gazebo/model_states",
                                                 ModelStates, self.callback_model_states)
        self.laser_sub = rospy.Subscriber("/" + name_space + "/ardrone/laser",
                                          LaserScan, self.callback_laser)
         
        rospy.loginfo("Subscribers initialized")
                                 
        # Publishers initialization
        self.model_state_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
        self.reset_pub = rospy.Publisher("/" + name_space + '/ardrone/reset', Empty, queue_size=10)
        self.takeoff_pub = rospy.Publisher("/" + name_space + '/ardrone/takeoff', Empty, queue_size=10)
        self.land_pub = rospy.Publisher("/" + name_space + '/ardrone/land', Empty, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher("/" + name_space + '/cmd_vel', Twist, queue_size=10)
    
        rospy.loginfo("Publishers initialized")
    
        # Small sleep to give time to publishers to open the topics
        rospy.sleep(0.5)     
    
        rospy.loginfo("taking off")
        self.takeoff_pub.publish(Empty())
                                          
        self.bridge = CvBridge()
        
        self.old_dist = 0.0
        actual_time = rospy.get_rostime()
        self.start_time =  actual_time.secs + \
                           actual_time.nsecs / 1000000000 
        self.old_time = self.start_time
                                          
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
                    
    def callback_image(self,data):
      self.img_msg = data
      self.img_flag = True
    
    def callback_bumper(self,data):                
      self.bumper_msg = data
      if self.bumper_msg.states != []:
          self.colliding_flag = True
      else:
          self.colliding_flag = False
              
        
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
        if data.name[i] == "quadrotor_" + str(self.number):
          is_there = i
    
      if is_there >= 0:
        self.model_states_pose_msg = data.pose[is_there]
      else:
        self.model_states_pose_msg = None

    def reward_calculation(self):
        # Removing inf from laser ranges
        if self.laser_msg != None:
          laser_ranges = np.asarray(self.laser_msg.ranges)
          laser_ranges[laser_ranges == inf] = 0
        
        # Calculating laser punishment
#        laser_cost = 0.0
#        if self.laser_msg != None:
#          inverted_ranges = 1 - (laser_ranges / self.laser_msg.range_max)
#          gaussian_ranges = np.multiply(inverted_ranges, signal.gaussian(self.feature_dim, (self.feature_dim / 2 * 0.8)))
#          laser_cost = -np.sum(gaussian_ranges) / self.feature_dim
#        rospy.logdebug("Laser range punishment: " + str(laser_cost))
        
        # Calculating the punishment for colliding
        is_colliding = False
        collision_cost = 0.0
        if self.colliding_flag:
            is_colliding = True
            collision_cost = -100.0
        rospy.logdebug("Collisions punishment: " + str(collision_cost))
            
        # Calculating the time elapsed from the last respawn
        actual_time = rospy.get_rostime()        
        time_stamp = actual_time.secs + \
                     actual_time.nsecs / 1000000000                    
        time_elapsed = time_stamp - self.start_time
        time_cost = -time_elapsed / 0.5
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
        trav_cost = 0.0
        alt_cost = 0.0
        if self.model_states_pose_msg != None:
          actual_pos = self.model_states_pose_msg.position
          trav_dist = np.sqrt((actual_pos.x - self.start_pos.linear.x)**2 + \
                              (actual_pos.y - self.start_pos.linear.y)**2)
          if (time_elapsed == 0):
              time_elapsed = 0.01
          #trav_cost = trav_dist / time_elapsed
          trav_cost = trav_dist
          alt_cost = -abs(1 - actual_pos.z)
        rospy.logdebug("Travel distance reward: " + str(trav_cost))
    
        # Calculating the angular velocity punishment
        angular_cost = 0
        if self.imu_msg != None:
            angular_cost = -abs(self.imu_msg[6])
        rospy.logdebug("Angular punishment: " + str(angular_cost))
        
        # Calculating the step reward
        if np.isnan(collision_cost):
            collision_cost = 0
        if np.isnan(aruco_cost):
            aruco_cost = 0
        if np.isnan(trav_cost):
            trav_cost = 0
        if np.isnan(alt_cost):
            alt_cost = 0
#        if np.isnan(laser_cost):
#            laser_cost = 0
        if np.isnan(angular_cost):
            angular_cost = 0 
        if np.isnan(time_cost):
            time_cost = 0    
            
        step_reward = collision_cost + \
                      (10 * aruco_cost) + \
                      trav_cost + alt_cost  + time_cost #\
                      #+ laser_cost + angular_cost
        rospy.logdebug("Step reward: " + str(step_reward))
        
        return step_reward, aruco_cost, is_colliding
        
    def reward_calculation2(self):
        # Removing inf from laser ranges
        if self.laser_msg != None:
          laser_ranges = np.asarray(self.laser_msg.ranges)
          laser_ranges[laser_ranges == inf] = 0
        
        # Calculating laser punishment
#        laser_cost = 0.0
#        if self.laser_msg != None:
#          inverted_ranges = 1 - (laser_ranges / self.laser_msg.range_max)
#          gaussian_ranges = np.multiply(inverted_ranges, signal.gaussian(self.feature_dim, (self.feature_dim / 2 * 0.8)))
#          laser_cost = -np.sum(gaussian_ranges) / self.feature_dim
#        rospy.logdebug("Laser range punishment: " + str(laser_cost))
        
        # Calculating the punishment for colliding
        is_colliding = False
        collision_cost = 0.0
        if self.colliding_flag:
            is_colliding = True
            collision_cost = -1.0
        rospy.logdebug("Collisions punishment: " + str(collision_cost))
            
        # Calculating the time elapsed from the last respawn
        actual_time = rospy.get_rostime()        
        time_stamp = actual_time.secs + \
                     actual_time.nsecs / 1000000000                    
        time_elapsed = time_stamp - self.old_time
        time_cost = -time_elapsed
        self.old_time = time_stamp
        rospy.logdebug("Time cost: " + str(time_cost))
        
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
        trav_cost = 0.0
        alt_cost = 0.0
        if self.model_states_pose_msg != None:
          actual_pos = self.model_states_pose_msg.position
          trav_dist = np.sqrt((actual_pos.x - self.start_pos.linear.x)**2 + \
                              (actual_pos.y - self.start_pos.linear.y)**2)
          trav_cost = trav_dist - self.old_dist
          alt_cost = -abs(1 - actual_pos.z)
          self.old_dist = trav_dist
        rospy.logdebug("Travel distance reward: " + str(trav_cost))
        rospy.logdebug("Altitude punishment: " + str(trav_cost))
        
    
        # Calculating the angular velocity punishment
        angular_cost = 0
        if self.imu_msg != None:
            angular_cost = -abs(self.imu_msg[6])
        rospy.logdebug("Angular punishment: " + str(angular_cost))        
        
        nan_rew = 0.0        
        # Calculating the step reward
        if np.isnan(collision_cost):
            print "Collision NAN!!!!. WORKER: " +  str(self.number)
            collision_cost = 0
            nan_rew = 1.0
        if np.isnan(aruco_cost):
            print "Aruco NAN!!!!. WORKER: " +  str(self.number)
            aruco_cost = 0
            nan_rew = 1.0
        if np.isnan(trav_cost):
            print "Trav NAN!!!!. WORKER: " +  str(self.number)
            trav_cost = 0
            nan_rew = 1.0
        if np.isnan(alt_cost):
            print "Alt NAN!!!!. WORKER: " +  str(self.number)
            alt_cost = 0
            nan_rew = 1.0
#        if np.isnan(laser_cost):
#            laser_cost = 0
        if np.isnan(angular_cost):
            print "Angular NAN!!!!. WORKER: " +  str(self.number)
            angular_cost = 0
            nan_rew = 1.0
        if np.isnan(time_cost):
            print "Time NAN!!!!. WORKER: " +  str(self.number)
            time_cost = 0
            nan_rew = 1.0
            
        step_reward = (10 * collision_cost) + \
                      trav_cost #+ alt_cost  + \
                      #(0.5 * time_cost) + \
                      #(10 * aruco_cost) + \
                      #+ laser_cost + angular_cost
        rospy.logdebug("Step reward: " + str(step_reward))
        
        return step_reward, aruco_cost, is_colliding, trav_cost, (0.5 * time_cost), nan_rew
        
    def train(self,rollout,sess,gamma,bootstrap_value):
        rospy.logdebug("Training")
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3]
        values = rollout[:,5]
        
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantagesFirst = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantagesFirst,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.inputs:np.vstack(observations),
            self.local_AC.actions:np.vstack(actions),
            self.local_AC.advantages:advantages,
            self.local_AC.state_in[0]:rnn_state[0],
            self.local_AC.state_in[1]:rnn_state[1]}
        v_l,p_l,e_l,g_n,v_n,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
              
        if math.isnan(e_l) or math.isnan(p_l) or math.isnan(v_l):
            print "STUCKKKKKKKKKKKKKKKKKKK"
            print "Entropy_loss: " + str(e_l) + \
                  ". Policy_loss: " + str(p_l) + \
                  ". Value_loss: " + str(v_l) + \
                  ". Inputs: " + str(np.vstack(observations)) + \
                  ". Actions: " + str(np.vstack(actions)) + \
                  ". Advantages: " + str(advantages) + \
                  ". State_in[0]: " + str(rnn_state[0]) + \
                  ". State_in[1]: " + str(rnn_state[1]) + \
                  ". Bootstrap_value: " + str([bootstrap_value]) + \
                  ". Rewards: " + str(rewards.tolist()) + \
                  ". Rewards_plus: " + str(self.rewards_plus) + \
                  ". Discounted_rewards: " + str(discounted_rewards) + \
                  ". Value_plus: " + str(self.value_plus) + \
                  ". AdvantagesFirst: " + str(advantagesFirst) + \
                  ". WORKER: " +  str(self.number)
            input()
            
                      
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n
        
    def work(self,max_episode_length,gamma,sess,coord,saver):            
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        rospy.loginfo("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                if self.img_flag:

                    # Reseting the episode starting time
                    actual_time = rospy.get_rostime()
                    self.start_time =  actual_time.secs + \
                                       actual_time.nsecs / 1000000000
                    self.old_time = self.start_time
                    self.old_dist = 0.0
                    
                    # initialization a3c variables
                    sess.run(self.update_local_ops)
                    episode_buffer = []
                    episode_values = []
                    episode_frames = []
                    episode_reward = 0
                    episode_step_count = 0
                    d = False                    
                    
                    # Reading the camera image from the topic
                    try:
                        cv_image = self.bridge.imgmsg_to_cv2(self.img_msg, "bgr8")
                        self.img_flag = False
                    except CvBridgeError as err:
                        rospy.logerror(err)
          
                    # Risizeing the image to 160x80 pixel for the convolutional network
                    #cv_image_resized = cv2.resize(cv_image, (160, 80))
                    #array_image = np.asarray(cv_image_resized)
                    array_image = np.asarray(cv_image, dtype='float32')
                    #input_image = np.zeros((1, 80, 160, 3))
                    #input_image[0] = array_image
                    
                    
                    # Laser sensor part !!!!!NEW!!!!!
                    laser_ranges = np.asarray(self.laser_msg.ranges)
                    laser_ranges[laser_ranges == inf] = 0.0
                    inverted_ranges = 1 - (laser_ranges / self.laser_msg.range_max)
                    array_image = inverted_ranges

                    
                    # process and store the frame
                    
                    
                    # LITTLE CHANGE
                    #episode_frames.append(array_image)
                    #input_image = process_frame(array_image)
                    episode_frames.append(array_image)
                    input_image = array_image
                    s = input_image
                    # rnn initialization
                    rnn_state = self.local_AC.state_init
                    
                    #Take an action using probabilities from policy network output.
                    a,v,rnn_state = sess.run([self.local_AC.samp_action,self.local_AC.value,self.local_AC.state_out], 
                                                 feed_dict={self.local_AC.inputs:[s],
                                                            self.local_AC.state_in[0]:rnn_state[0],
                                                            self.local_AC.state_in[1]:rnn_state[1]})
                    
                    #print "ACTIOOOONNNNSSSS BEFOREEEE: " + str(a)
                    a = np.squeeze(a)
                    
                    
                    # Perform an action
                    cmd_input = Twist()
                    cmd_input.linear.x = 0.5#a[0]
                    cmd_input.linear.y = 0.0#a[1]
                    cmd_input.linear.z = 0.0#a[2]
                    cmd_input.angular.z = a[1]#a[3]
                    self.cmd_vel_pub.publish(cmd_input)
                    self.img_flag = False
                                         
                    travSum = 0.0
                    timeSum = 0.0            
                    while d == False:
                        if self.img_flag:
                            #reward version 1
                            #r, aruco_cost, is_colliding = self.reward_calculation()
                            #reward version 2
                            r, aruco_cost, is_colliding, trav, timeCost, nan_rew = self.reward_calculation2()
                            travSum += trav
                            timeSum += timeCost
                            
                            if (is_colliding and self.navdata_msg.state == 3) or \
                               (aruco_cost > 0.8) or \
                               episode_step_count == max_episode_length - 1:
                                d = True
                            else:  
                                d = False                            
                            
                            if d == False:
                                # Reading the camera image from the topic
                                try:
                                    cv_image = self.bridge.imgmsg_to_cv2(self.img_msg, "bgr8")
                                    self.img_flag = False
                                except CvBridgeError as err:
                                    rospy.logerror(err)
                                    
                                # Laser sensor part !!!!!NEW!!!!!
                                laser_ranges = np.asarray(self.laser_msg.ranges)
                                laser_ranges[laser_ranges == inf] = 0.0
                                inverted_ranges = 1 - (laser_ranges / self.laser_msg.range_max)
                                array_image = inverted_ranges
                                
                                # Risizeing the image to 160x80 pixel for the convolutional network
                                #array_image = np.asarray(cv_image, dtype='float32')
                                #episode_frames.append(array_image)
                                #input_image = process_frame(array_image)
                                episode_frames.append(array_image)
                                input_image = array_image
                                s1 = input_image
                            else:
                                s1 = s
                                
                            episode_buffer.append([s,a,r,s1,d,v[0,0]])
                            episode_values.append(v[0,0])
        
                            episode_reward += r
                            s = s1                    
                            total_steps += 1
                            episode_step_count += 1
                            
                            if nan_rew:
                                print "Episode: " + str(episode_count) + ". Step: " + str(total_steps)
                            
                            # If the episode hasn't ended, but the experience buffer is full, then we
                            # make an update step using that experience rollout.
                            if len(episode_buffer) == 30 and d != True and episode_step_count != max_episode_length - 1:
                                # Since we don't know what the true final return is, we "bootstrap" from our current
                                # value estimation. 
                                v1 = sess.run(self.local_AC.value, 
                                     feed_dict={self.local_AC.inputs:[s],
                                     self.local_AC.state_in[0]:rnn_state[0],
                                     self.local_AC.state_in[1]:rnn_state[1]})[0,0]
                                while np.isnan(v1):
                                    print "value bootstrap is NaN. WORKER: " +  str(self.number)
                                    print "Episode: " + str(episode_count) + ". Step: " + str(total_steps)
                                    v1 = 0.0
#                                    v1 = sess.run(self.local_AC.value, 
#                                         feed_dict={self.local_AC.inputs:[s],
#                                         self.local_AC.state_in[0]:rnn_state[0],
#                                         self.local_AC.state_in[1]:rnn_state[1]})[0,0]    
                                v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,v1)
                                episode_buffer = []
                                sess.run(self.update_local_ops)
                            if d == True:
                                break
                            
                            #Take an action using probabilities from policy network output.
                            a,v,rnn_state = sess.run([self.local_AC.samp_action,self.local_AC.value,self.local_AC.state_out], 
                                                        feed_dict={self.local_AC.inputs:[s],
                                                                   self.local_AC.state_in[0]:rnn_state[0],
                                                                   self.local_AC.state_in[1]:rnn_state[1]})         
                                                                   
                            #print "ACTIOOOONNNNSSSS BEFOREEEE: " + str(a)
                            a = np.squeeze(a)
                            
                            # Perform an action
                            cmd_input = Twist()
                            cmd_input.linear.x = 0.5#a[0]
                            cmd_input.linear.y = 0.0#a[1]
                            cmd_input.linear.z = 0.0#a[2]
                            cmd_input.angular.z = a[1]#a[3]
                            self.cmd_vel_pub.publish(cmd_input)
                            self.img_flag = False
                                    
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_step_count)
                    self.episode_mean_values.append(np.mean(episode_values))
                    
                    # Update the network using the experience buffer at the end of the episode.
                    if len(episode_buffer) != 0:
                        v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)
                    
#                    print "Entropy_loss: " + str(e_l) + \
#                          ". Policy_loss: " + str(p_l) + \
#                          ". Value_loss: " + str(v_l) + \
#                          ". Episodes: " + str(episode_count) + \
#                          ". Trav_cost: " + str(travSum) + \
#                          ". Time_cost: " + str(timeSum) + \
#                          ". WORKER: " +  str(self.number)
                        
                    # Periodically save gifs of episodes, model parameters, and summary statistics.
                    if episode_count % 5 == 0 and episode_count != 0:
#                        if self.name == 'worker_0' and episode_count % 25 == 0:
#                            time_per_step = 0.05
#                            images = np.array(episode_frames)
#                            make_gif(images,self.net_path + "/frames/image"+str(episode_count)+'.gif',
#                                duration=len(images)*time_per_step,true_image=True,salience=False)
                        if episode_count % 25 == 0 and self.name == 'worker_0':
                            saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                            print "Saved Model"
    
                        mean_reward = np.mean(self.episode_rewards[-5:])
                        mean_length = np.mean(self.episode_lengths[-5:])
                        mean_value = np.mean(self.episode_mean_values[-5:])
                        summary = tf.Summary()
                        summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                        summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                        summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                        self.summary_writer.add_summary(summary, episode_count)
    
                        self.summary_writer.flush()
                    if self.name == 'worker_0':
                        sess.run(self.increment)
                    episode_count += 1
                    
                    model_state_msg = ModelState()
                    empty_msg = Empty()
                    
                    # Generating a new random position chosen between 4
                    new_position = random.sample(self.position_list,  1)
                    
                    # Generating a random orientation
                    angle = random.random() * 2 * np.pi
                    
#                    rospy.loginfo("New position: " + str(new_position) + \
#                                  "New angle: " + str(angle))
                    
                    # Creating the model state message to send to set_model_space topic
                    model_state_msg.model_name = self.quadrotor_name
                    model_state_msg.pose.position.x = new_position[0][0]
                    model_state_msg.pose.position.y = new_position[0][1] + (self.world_step * (self.num_drone - 1))
                    model_state_msg.pose.position.z = 0.5
                    quaternion = quaternion_from_euler(0, 0, angle)
                    model_state_msg.pose.orientation.x = quaternion[0]
                    model_state_msg.pose.orientation.y = quaternion[1]
                    model_state_msg.pose.orientation.z = quaternion[2]
                    model_state_msg.pose.orientation.w = quaternion[3]
                    model_state_msg.reference_frame = "world"
                    
                    self.start_pos.linear.x = model_state_msg.pose.position.x
                    self.start_pos.linear.y = model_state_msg.pose.position.y
                    self.start_pos.linear.z = model_state_msg.pose.position.z
                                       
                 
                    # Reinitializing position, orientation and status of the drone
                    self.land_pub.publish(empty_msg)
                    
                    # reset the actions
                    delta_t = 0.01
                    wait_time = 0.5
                    for it in range (int(wait_time / delta_t)):
                        cmd_input.linear.x = 0.0
                        cmd_input.linear.y = 0.0
                        cmd_input.linear.z = 0.0
                        cmd_input.angular.z = 0.0
                        self.cmd_vel_pub.publish(cmd_input)
                        rospy.sleep(delta_t)                    
                    
                    self.model_state_pub.publish(model_state_msg)
                    self.reset_pub.publish(empty_msg)
                    rospy.sleep(0.5) 
                    self.takeoff_pub.publish(empty_msg)
                                        
                    # reset the actions
                    delta_t = 0.01
                    wait_time = 0.5
                    for it in range (int(wait_time / delta_t)):
                        cmd_input.linear.x = 0.0
                        cmd_input.linear.y = 0.0
                        cmd_input.linear.z = 0.0
                        cmd_input.angular.z = 0.0
                        self.cmd_vel_pub.publish(cmd_input)
                        rospy.sleep(delta_t)
                    
                    actual_time = rospy.get_rostime()
                    self.start_time =  actual_time.secs + \
                                       actual_time.nsecs / 1000000000                     
                    
                    self.colliding_flag = False