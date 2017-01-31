#!/usr/bin/env python
from __future__ import print_function, division

import roslib
import tf
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

class actor_critic:

  def __init__(self):     
    random.seed()
      
    self.aruco_limit = 4.0
    self.bumper_msg = None
    self.navdata_msg = None
    self.aruco_msg = None
    self.model_states_pose_msg = None    
    self.count = 0;
    self.bridge = CvBridge() 
    
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
        time_stamp = data.header.stamp.secs + \
                     data.header.stamp.nsecs / 1000000000
        
        # Reading the camera image from the topic
        try:
          cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          rospy.logerror(e)
        
        # Risizeing the image to 160x80 pixel for the convolutional network
        cv_image_resized = cv2.resize(cv_image, (160, 80))

        # Calculating the aruco distance reward
        aruco_dist = 0.0        
        if self.aruco_msg != None:
            aruco_dist = np.sqrt(self.aruco_msg.linear.x**2 + \
                                 self.aruco_msg.linear.y**2 + \
                                 self.aruco_msg.linear.z**2)
            if aruco_dist == 0.0 or aruco_dist > self.aruco_limit:
                aruco_dist = self.aruco_limit
            aruco_dist = 1.0 - (aruco_dist / self.aruco_limit)
            rospy.logdebug("Aruco distance reward: " + str(aruco_dist))
            
        # Calculating the traveled distance reward
        trav_dist = 0.0
        if self.model_states_pose_msg != None:
          actual_pos = self.model_states_pose_msg.position
          trav_dist = np.sqrt((actual_pos.x - self.start_pos.linear.x)**2 + \
                              (actual_pos.y - self.start_pos.linear.y)**2 + \
                              (actual_pos.z - self.start_pos.linear.z)**2)
        rospy.loginfo("Travel distance reward: " + str(trav_dist))
        # IS MISSING THE CALCULATION OF TIME reward = trav_dist - time_elapsed
        
        # Starting a newe episode if the drone collide with something or if it's close enough to an aruco board
        if (self.bumper_msg != None  and \
            self.bumper_msg.states != []) or \
            (aruco_dist > 0.8):
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

  def callback_imu(self,data):
    time_stamp = data.header.stamp.secs + \
                 data.header.stamp.nsecs / 1000000000
                 
    info = str(data.orientation.x) + " "  + \
           str(data.orientation.y) + " "  + \
           str(data.orientation.z) + " "  + \
           str(data.orientation.w) + " "  + \
           str(data.angular_velocity.x) + " "  + \
           str(data.angular_velocity.y) + " "  + \
           str(data.angular_velocity.z) + " "  + \
           str(data.linear_acceleration.x) + " "  + \
           str(data.linear_acceleration.y) + " "  + \
           str(data.linear_acceleration.z) + " "  + \
           str(data.orientation_covariance[0]) + " "  + \
           str(data.orientation_covariance[1]) + " "  + \
           str(data.orientation_covariance[2]) + " "  + \
           str(data.orientation_covariance[3]) + " "  + \
           str(data.orientation_covariance[4]) + " "  + \
           str(data.orientation_covariance[5]) + " "  + \
           str(data.orientation_covariance[6]) + " "  + \
           str(data.orientation_covariance[7]) + " "  + \
           str(data.orientation_covariance[8]) + " "  + \
           str(data.angular_velocity_covariance[0]) + " "  + \
           str(data.angular_velocity_covariance[1]) + " "  + \
           str(data.angular_velocity_covariance[2]) + " "  + \
           str(data.angular_velocity_covariance[3]) + " "  + \
           str(data.angular_velocity_covariance[4]) + " "  + \
           str(data.angular_velocity_covariance[5]) + " "  + \
           str(data.angular_velocity_covariance[6]) + " "  + \
           str(data.angular_velocity_covariance[7]) + " "  + \
           str(data.angular_velocity_covariance[8]) + " "  + \
           str(data.linear_acceleration_covariance[0]) + " "  + \
           str(data.linear_acceleration_covariance[1]) + " "  + \
           str(data.linear_acceleration_covariance[2]) + " "  + \
           str(data.linear_acceleration_covariance[3]) + " "  + \
           str(data.linear_acceleration_covariance[4]) + " "  + \
           str(data.linear_acceleration_covariance[5]) + " "  + \
           str(data.linear_acceleration_covariance[6]) + " "  + \
           str(data.linear_acceleration_covariance[7]) + " "  + \
           str(data.linear_acceleration_covariance[8]) + " "  + \
           str(time_stamp) + "\n"
    
    rospy.logdebug(info)
    
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
