#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('opticalflow_controller')
import sys
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class optical_flow:

  def __init__(self):
    self.first = True
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/vizzy/r_camera/image_rect_color", Image, self.callback)

    print("Subscriber initialized")

  def draw_flow(self, img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    #variables for the mean
    mean_l_x = 0
    mean_l_y = 0
    mean_l_vx = 0
    mean_l_vy = 0
    total_l = 0
    mean_r_x = 0
    mean_r_y = 0
    mean_r_vx = 0
    mean_r_vy = 0
    total_r = 0
    for (x1, y1), (x2, y2) in lines:
      cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
      #checks for the mean
      if x1 < w / 2:
        total_l += 1
        mean_l_x += x1
        mean_l_y += y1
        mean_l_vx += x2 - x1
        mean_l_vy += y2 - y1
      else:
        total_r += 1
        mean_r_x += x1
        mean_r_y += y1
        mean_r_vx += x2 - x1
        mean_r_vy += y2 - y1
    #calculating the mean
    if total_l > 0:
      mean_l_x /= total_l
      mean_l_y /= total_l
      mean_l_vx /= total_l
      mean_l_vy /= total_l
    if total_r > 0:
      mean_r_x /= total_r
      mean_r_y /= total_r
      mean_r_vx /= total_r
      mean_r_vy /= total_r
    print(mean_l_x)
    print(mean_l_y)
    print("------")
    size_l = np.sqrt(pow(mean_l_vx, 2) + pow(mean_l_vy, 2))
    size_r = np.sqrt(pow(mean_r_vx, 2) + pow(mean_r_vy, 2))
    cv2.line(vis, (mean_l_x, mean_l_y),
            (mean_l_x + mean_l_vx, mean_l_y + mean_l_vy),
            (255, 0, 0), 1, 8, 0)
    cv2.circle(vis, (mean_l_x, mean_l_y), 1, (255, 0, 0), -1)
    cv2.line(vis, (mean_r_x, mean_r_y),
            (mean_r_x + mean_r_vx, mean_r_y + mean_r_vy),
            (255, 0, 0), 1, 8, 0)
    cv2.circle(vis, (mean_r_x, mean_r_y), 1, (255, 0, 0), -1)
    return vis

  def draw_hsv(self, flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

  def warp_flow(self, img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    if self.first:
      self.prevgray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
      self.first = False
    else:
      gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
      flow = cv2.calcOpticalFlowFarneback(self.prevgray, gray, 0.5, 3, 15, 3, 5, 1.2, 0)
      self.prevgray = gray
      cv2.imshow("Image window", self.draw_flow(gray, flow))
      #if self.show_hsv:
        #cv2.imshow('flow HSV', self.draw_hsv(flow))
      #if self.show_glitch:
        #self.cur_glitch = self.warp_flow(self.cur_glitch, flow)
        #cv2.imshow('glitch', self.cur_glitch)

def main(args):
  cv2.startWindowThread()
  cv2.namedWindow("Image window")
  of = optical_flow()
  rospy.init_node('optical_flow', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)