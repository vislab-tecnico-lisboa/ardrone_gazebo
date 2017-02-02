#!/usr/bin/env python
from __future__ import print_function, division

import roslib
import random
roslib.load_manifest('actor_critic_agent')
import sys
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState, ModelStates, ContactsState
from ardrone_autonomy.msg import Navdata
from std_msgs.msg import Empty
from cv_bridge import CvBridge, CvBridgeError
import time
import os
import matplotlib.pyplot as plt

from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img
from autoencoder import autoencoder_network
import tensorflow as tf
from collections import deque


class actor_critic:

  def __init__(self):     
    random.seed()
    
    self.queue = deque([])
    
    self.graph = tf.get_default_graph()    
    
    with self.graph.as_default():
      self.autoencoder_network = autoencoder_network()
      
    self.aruco_limit = 4.0
    self.imu_msg = np.zeros(37)
    self.bumper_msg = None
    self.navdata_msg = None
    self.aruco_msg = None
    self.model_states_pose_msg = None    
    self.count = 0;
    self.bridge = CvBridge()

    actual_time = rospy.get_rostime()
    self.start_time =  actual_time.secs + \
                       actual_time.nsecs / 1000000000
    
    self.start_pos = Twist()
    self.start_pos.linear.x = 0.0
    self.start_pos.linear.y = 0.0
    self.start_pos.linear.z = 0.0
    
    self.total_reward = 0.0
    
    self.position_list = [[0, 0], [20, 0], [20, -20], [0, -20]]
    
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
                                      
    rospy.loginfo("Subscribers initialized")
    
    # Publishers initialization
    self.model_state_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
    self.reset_pub = rospy.Publisher('/ardrone/reset', Empty, queue_size=10)
    self.takeoff_pub = rospy.Publisher('/ardrone/takeoff', Empty, queue_size=10)
    self.land_pub = rospy.Publisher('/ardrone/land', Empty, queue_size=10)
    
    rospy.loginfo("Publishers initialized")
    
    # Small sleep to give time to publishers to open the topics
    rospy.sleep(0.5)     
    
    rospy.loginfo("taking off")
    self.takeoff_pub.publish(Empty())

  def callback_image(self,data):
    if (self.navdata_msg != None and \
        self.navdata_msg.state == 3):  # The actor-critic system works only when the drone is flying
        # Calculating the punishment for colliding
        is_colliding = False
        collision_cost = 0.0
        if self.bumper_msg != None  and \
           self.bumper_msg.states != []:
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
            
        # Calculating the traveled distance reward
        trav_dist = 0.0
        if self.model_states_pose_msg != None:
          actual_pos = self.model_states_pose_msg.position
          trav_dist = np.sqrt((actual_pos.x - self.start_pos.linear.x)**2 + \
                              (actual_pos.y - self.start_pos.linear.y)**2)
          trav_cost = trav_dist / time_elapsed
        rospy.logdebug("Travel distance reward: " + str(trav_cost))
        
        # Calculating the step reward
        step_reward = collision_cost + aruco_cost + trav_cost
        rospy.logdebug("Step reward: " + str(step_reward))
        
        # Calculating the total reward
        self.total_reward += step_reward
        rospy.logdebug("total reward: " + str(self.total_reward))
        
        # Reading the camera image from the topic
        try:
          cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          rospy.logerror(e)
        
        # Risizeing the image to 160x80 pixel for the convolutional network
        cv_image_resized = cv2.resize(cv_image, (160, 80))
        #array_image = np.asarray(cv_image_resized)
        array_image = img_to_array(cv_image_resized)
        input_image = np.zeros((1, 80, 160, 3))
        input_image[0] = array_image
        
        # Calculation the image features
        with self.graph.as_default():
          image_features = self.autoencoder_network.run_network(input_image)
          
        # adding the features to the features list
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
        state = np.concatenate((self.queue[0].flatten(), 
                                self.queue[1].flatten(),
                                self.queue[2].flatten(),
                                self.imu_msg))
        rospy.logdebug("State shape: " + str(len(state)))
        
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
            quaternion = tf.transformations.quaternion_from_euler(0, 0, angle)
            model_state_msg.pose.orientation.x = quaternion[0]
            model_state_msg.pose.orientation.y = quaternion[1]
            model_state_msg.pose.orientation.z = quaternion[2]
            model_state_msg.pose.orientation.w = quaternion[3]
            model_state_msg.reference_frame = "world"
            
            self.start_pos.linear.x = new_position[0][0]
            self.start_pos.linear.y = new_position[0][1]
            self.start_pos.linear.z = 0.0
    
            # Reinitializing position, orientation and status of the drone
            self.land_pub.publish(empty_msg)
            self.model_state_pub.publish(model_state_msg)
            self.reset_pub.publish(empty_msg)
            rospy.sleep(0.5)
            self.takeoff_pub.publish(empty_msg)
            
            # Reseting the episode starting time
            actual_time = rospy.get_rostime()
            self.start_time =  actual_time.secs + \
                               actual_time.nsecs / 1000000000
            self.total_reward = 0.0
            # Reseting the image queue
            self.queue = deque([])

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
    linear_acceleration[0] = data.linear_acceleration.x
    linear_acceleration[1] = data.linear_acceleration.y
    linear_acceleration[2] = data.linear_acceleration.z
    
    self.imu_msg = np.concatenate((orientation, 
                                   angular_velocity,
                                   linear_acceleration,
                                   data.orientation_covariance,
                                   data.angular_velocity_covariance,
                                   data.linear_acceleration_covariance))
    
  def callback_bumper(self,data):                 
    self.bumper_msg = data
    
  def callback_navdata(self,data): 
    # 0: Unknown, 1: Init, 2: Landed, 3: Flying, 4: Hovering, 5: Test
    # 6: Taking off, 7: Goto Fix Point, 8: Landing, 9: Looping
    # Note: 3,7 seems to discriminate type of flying (isFly = 3 | 7)                 
    self.navdata_msg = data
    
  def callback_aruco(self,data):                 
    self.aruco_msg = data
    
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
  rospy.init_node('optical_flow', anonymous=True, log_level=rospy.INFO)
  
  ac = actor_critic()
  time.sleep(1)
  
  rospy.loginfo("<------Data recorder (Author: Nino Cauli)------>")
  
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")   
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
