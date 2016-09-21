#!/usr/bin/env python
from __future__ import print_function, division

import roslib
roslib.load_manifest('opticalflow_controller')
import sys
import rospy
import numpy as np
import cv2
from common import draw_str
from sensor_msgs.msg import Image, Imu, CameraInfo
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from cv_bridge import CvBridge, CvBridgeError
from keypressed import getchar
import time

lk_params = dict(winSize  = (15, 15),
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners = 500,
                      qualityLevel = 0.05,
                      minDistance = 7,
                      blockSize = 7)

front_time = 1000
turn_time = 500

class optical_flow:

  def __init__(self):
    self.past_time = int(round(time.time() * 1000))
    self.turn_flag = False
    self.print_flag = True
    self.ctrl_flag = False
    self.rot_rem_flag = True
    self.counter = -1
    
    self.old_time_stamp = 0

    self.track_len = 10
    self.detect_interval = 5
    self.tracks = []
    self.tracks_norot = []
    self.frame_idx = 0
    self.bridge = CvBridge()
    self.imu_data = []
    self.image_info = CameraInfo()
    self.info_ready = False
    self.imu_ready = False
    self.image_sub = rospy.Subscriber("/ardrone/front/image_raw",
                                      Image, self.callback_image)
    self.info_sub = rospy.Subscriber("/ardrone/front/camera_info",
                                      CameraInfo, self.callback_info)
    self.imu_sub = rospy.Subscriber("/ardrone/imu",
                                      Imu, self.callback_imu)
    print("Subscribers initialized")

    self.ctrl_pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 1)
    self.takeoff_pub  = rospy.Publisher('/ardrone/takeoff', Empty, queue_size = 1)
    self.land_pub  = rospy.Publisher('/ardrone/land', Empty, queue_size = 1)
    print("Publishers initialized")      
  
  def landing(self):
    print("Landing!")
    self.land_pub.publish(Empty())  
  
  def taking_off(self):          
    print("Taking off!")
    self.takeoff_pub.publish(Empty())
    
  def print_on_off(self):
    if self.print_flag:          
      print("printing off!")
      self.print_flag = False
    else:
      print("printing on!")
      self.print_flag = True
      
  def ctrl_on_off(self):
    if self.ctrl_flag:          
      print("controller off!")
      self.ctrl_flag = False
    else:
      print("controller on!")
      self.ctrl_flag = True
      
  def rot_rem_on_off(self):
    if self.rot_rem_flag:          
      print("Rotation correction removed")
      self.rot_rem_flag = False
    else:
      print("Rotation correction added")
      self.rot_rem_flag = True


  def reactive_controller(self, size_l, size_r):
    twist = Twist()
    l_r_sum = size_l + size_r
    l_r_sum_abs = abs(size_l) + abs(size_r)

    if l_r_sum_abs >= 50:
        twist.linear.x = 0
        twist.angular.z = 0.0
        #print('Stop: ', l_r_sum, l_r_sum_abs, size_l, size_r)
    elif abs(l_r_sum) <= (l_r_sum_abs * 20 / 100):
        twist.linear.x = 1
        twist.angular.z = 0.0
        #print('Forward: ', l_r_sum, l_r_sum_abs, size_l, size_r)
    else:
        twist.linear.x = 1
        twist.angular.z = l_r_sum / l_r_sum_abs
        #print('Turn: ', l_r_sum, l_r_sum_abs, size_l, size_r)   
    
    return twist

  def reactive_controller2(self, size_l, size_r):
    twist = Twist()
    l_r_sum = size_l + size_r
    l_r_sum_abs = abs(size_l) + abs(size_r)

    if ((abs(l_r_sum) <= (l_r_sum_abs * 50 / 100)) or (self.counter <= front_time)) and (self.turn_flag == False):
#    if ((abs(l_r_sum) <= 0.5) or (self.counter <= front_time)) and (self.turn_flag == False):
      if self.counter == -1:
        self.past_time = int(round(time.time() * 1000))
      time_now = int(round(time.time() * 1000))
      self.counter = time_now - self.past_time
      twist.linear.x = 3.00
      twist.angular.z = 0.0
    else:
      if self.turn_flag == False:
        self.past_time = int(round(time.time() * 1000))
        self.turn_flag = True
      self.counter = int(round(time.time() * 1000)) - self.past_time
      if self.counter > turn_time:
        self.turn_flag = False
        self.counter = -1
      twist.linear.x = 0.00
      twist.angular.z = np.sign(l_r_sum) * 1.00
      #print('Turn: ', self.counter)
      #twist.linear.x = 0.50
      #twist.angular.z = l_r_sum / 20

    return twist

  def calc_mean(self, img, p0, p1, good):
    h, w = img.shape[:2]
    #vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = img.copy()
    #variables for the mean
    mean_l_x = 0
    mean_l_y = 0
    mean_l_vx = 0
    mean_l_vy = 0
    total_l = 0
    size_l = 0
    mean_r_x = 0
    mean_r_y = 0
    mean_r_vx = 0
    mean_r_vy = 0
    total_r = 0
    size_r = 0
    for (x1, y1), (x2, y2), good_flag in zip(np.int32(p0),
                                             np.int32(p1), good):
      #checks for the mean
      if good_flag:
        if x1 < w / 4:                     
          total_l += 1
          mean_l_x += x1
          mean_l_y += y1
          mean_l_vx += x2 - x1
          mean_l_vy += y2 - y1
          size_l += np.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))
        elif x1 > w * 3 / 4:
          total_r += 1
          mean_r_x += x1
          mean_r_y += y1
          mean_r_vx += x2 - x1
          mean_r_vy += y2 - y1
          size_r += np.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))
    #calculating the mean
    if total_l > 0:
      mean_l_x /= total_l
      mean_l_y /= total_l
      mean_l_vx /= total_l
      mean_l_vy /= total_l
      size_l = size_l / total_l * np.sign(mean_l_vx)
    if total_r > 0:
      mean_r_x /= total_r
      mean_r_y /= total_r
      mean_r_vx /= total_r
      mean_r_vy /= total_r
      size_r = size_r / total_r * np.sign(mean_r_vx)
    
    if self.print_flag:
      cv2.line(vis, (np.int32(w / 4), np.int32(h / 2)),
              (np.int32(w / 4 + size_l), np.int32(h / 2)),
              (0, 0, 255), 1, 8, 0)
      cv2.circle(vis, (np.int32(w / 4 + size_l), np.int32(h / 2)),
                 2, (0, 0, 255), -1)
      cv2.line(vis, (np.int32(w * 3 / 4), np.int32(h / 2)),
              (np.int32(w * 3 / 4 + size_r), np.int32(h / 2)),
              (0, 0, 255), 1, 8, 0)
      cv2.circle(vis, (np.int32(w * 3 / 4 + size_r), np.int32(h / 2)),
                 2, (0, 0, 255), -1)
    return vis, size_l, size_r
    
  def new_ending_point(self, x, y, imu_x, imu_y, imu_z, K, deltaT):
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    #print(imu_temp.angular_velocity.x, fx, fy, cx, cy)
    
    #yaw
    alpha = imu_z * deltaT   
    beta = np.arctan2(x - cx, fx)
    gamma = beta - alpha
    
    #print("Angles: ", alpha, beta, gamma)
    #print("x - cx: ", x - cx, "arctan2", beta)

    newx = cx + (fx * np.tan(gamma))
    
    #print("newx: ", newx)
    
    #pitch
    alpha = imu_y * deltaT   
    beta = np.arctan2(y - cy, fy)
    gamma = beta + alpha

    newy = cy + (fy * np.tan(gamma))
    
    #roll
    alpha = imu_x * deltaT   
    beta = np.arctan2(newy - cy, newx - cx)
    gamma = beta + alpha
    
    len_diag = np.sqrt(pow(newx - cx, 2) + pow(newy - cy, 2))
    
    newx = cx + (len_diag * np.cos(gamma))
    newy = cy + (len_diag * np.sin(gamma))
    
    return newx, newy
  
  def new_ending_point2(self, x, y, imu_x, imu_y, imu_z, K, deltaT):
    yaw = imu_x * deltaT
    pitch = -imu_z * deltaT 
    roll = -imu_y * deltaT
  
    rot_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], 
                        [np.sin(yaw), np.cos(yaw), 0], 
                        [0, 0, 1]])
                     
    rot_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)], 
                          [0, 1, 0],
                          [-np.sin(pitch), 0, np.cos(pitch)]])
                          
    rot_roll = np.array([[1, 0, 0],
                         [0, np.cos(roll), -np.sin(roll)], 
                         [0, np.sin(roll), np.cos(roll)]])
                         
    roll_tot = rot_yaw.dot(rot_pitch).dot(rot_roll)   
    
    T = np.concatenate((roll_tot, np.zeros((3, 1))), axis = 1)
    
    old_pos = np.array([[x], [y], [1]])
    
    point_3d_3dim = np.dot(np.linalg.pinv(K), old_pos)    
    point_3d = np.concatenate((point_3d_3dim, np.array([[1]])), axis = 0)
    
    new_pos = K.dot(T).dot(point_3d)    
    
    newx = new_pos[0]
    newy = new_pos[1]   
    
    return newx, newy

  def callback_image(self,data):
      
    if self.info_ready and self.imu_ready:
        imu_x = 0 
        imu_y = 0
        imu_z = 0
        
        imu_deltaT = 0
        
        time_stamp = data.header.stamp.secs + \
                     data.header.stamp.nsecs / 1000000000
        while len(self.imu_data) > 0:
            imu_time_stamp = self.imu_data[0].header.stamp.secs + \
                             self.imu_data[0].header.stamp.nsecs / 1000000000
            if imu_time_stamp <= time_stamp:
                imu_temp = self.imu_data.pop(0)
                imu_deltaT = 1 / (imu_temp.header.seq / imu_temp.header.stamp.secs)
                imu_x += imu_temp.angular_velocity.x
                imu_y += imu_temp.angular_velocity.y
                imu_z += imu_temp.angular_velocity.z
            else:
                break
                
        K = np.float32(self.info_data.K).reshape(3, 3)
        deltaT = time_stamp - self.old_time_stamp
        self.old_time_stamp = time_stamp
        
        #print("TS: ", deltaT, imu_deltaT)        
        
        
        try:
          cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          print(e)
    
        frame_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        vis = cv_image.copy()
    
        if len(self.tracks) > 0:
            img0, img1 = self.prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []
            new_tracks_norot = []
            start_p = []
            end_p = []
            for tr, tr_norot, (x, y), good_flag in zip(self.tracks, self.tracks_norot, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                #nino part
                #removing rotations
                (xold, yold) = np.float32(tr[-1])
                if self.rot_rem_flag:
                  (xold2, yold2) = np.float32(tr_norot[-1])
                  xrot, yrot = self.new_ending_point(xold, yold, imu_x, imu_y, imu_z, K, imu_deltaT)
                  x1 = np.float32((x - xold) + (xrot - xold) + xold2)
                  y1 = np.float32((y - yold) + (yrot - yold) + yold2)
                  #x1 = np.float32(x + (xrot - xold))
                  #y1 = np.float32(y + (yrot - yold))
                  tr_norot.append((x1, y1))
                tr.append((x, y))
                if len(tr) > self.track_len:
                  if self.rot_rem_flag:
                    del tr_norot[0]
                  del tr[0]
                #nino part   
                #mean vectors
                start_p.append((xold, yold))
                if self.rot_rem_flag:
                  #start_p.append(tr_norot[0])
                  #end_p.append((x1, y1))
                  (sp_x_old, sp_y_old) = np.float32(tr_norot[0])
                  ep_x = xold + (x1 - sp_x_old)
                  ep_y = yold + (y1 - sp_y_old)
                  
                else:
                  #start_p.append(tr[0])
                  #end_p.append((x, y))
                  (sp_x_old, sp_y_old) = np.float32(tr[0])
                  ep_x = xold + (x - sp_x_old)
                  ep_y = yold + (y - sp_y_old)
                end_p.append((ep_x, ep_y))

                if self.rot_rem_flag:
                  new_tracks_norot.append(tr_norot)
                new_tracks.append(tr)
                
                if self.print_flag:
#                  if self.rot_rem_flag:
#                    #Nino's print
#                    cv2.circle(vis, (x1, y1), 2, (0, 255, 0), -1)
#                  else:
#                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                  cv2.circle(vis, (ep_x, ep_y), 2, (0, 255, 0), -1)
                  cv2.line(vis, (xold, yold), (ep_x, ep_y), (0, 255, 0), 1, 8, 0)
            if self.rot_rem_flag:
              self.tracks_norot = new_tracks_norot
            self.tracks = new_tracks
            
#            if self.print_flag:
#              if self.rot_rem_flag:
#                #Nino's print
#                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks_norot], False, (0, 255, 0))
#              else:
#                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
#              draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))
            
            if self.ctrl_flag:
              vis, size_l, size_r = self.calc_mean(vis, start_p, end_p, good)            
              if self.print_flag:             
                draw_str(vis, (20, 40), 'Lenght left: %f' % size_l)
                draw_str(vis, (20, 60), 'Lenght right: %f' % size_r)
            
              self.ctrl_pub.publish(self.reactive_controller2(size_l, size_r))
    
        if self.frame_idx % self.detect_interval == 0:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                cv2.circle(mask, (x, y), 5, 0, -1)
            p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    self.tracks.append([(x, y)])
                    self.tracks_norot.append([(x, y)])
    
    
        self.frame_idx += 1
        self.prev_gray = frame_gray
    
        cv2.imshow("Image window", vis)
        cv2.waitKey(1)

  def callback_imu(self,data):
    self.imu_data.append(data)
    self.imu_ready = True
    
  def callback_info(self,data):
    self.info_data = data
    self.info_ready = True


def main(args):
  #cv2.startWindowThread()
  #cv2.namedWindow("Image window")
  rospy.init_node('optical_flow', anonymous=True)
  
  of = optical_flow()
  time.sleep(1)
  #of.taking_off()
  
  key = ''  
  
  while not rospy.is_shutdown():
      key = getchar()
      if key == 'q':
          break
      if key == 't':
          of.taking_off()
      if key == 'l':
          of.landing()
      if key == 'p':
          of.print_on_off()
      if key == 'c':
          of.ctrl_on_off()
      if key == 'r':
          of.rot_rem_on_off()
  
  print("Shutting down")    
  of.landing()    
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
