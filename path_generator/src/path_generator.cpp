#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "path_generator");
  
  ros::NodeHandle n;
  ros::Publisher path_pub = n.advertise<nav_msgs::Path>("path_pub", 1000);
  ros::Publisher pos_pub = n.advertise<geometry_msgs::PoseStamped>("pos_pub", 1000);

  int freq = 100;
  float round_min = 0.25;
  ros::Rate loop_rate(freq);
  
  nav_msgs::Path path_msg;
  path_msg.poses = {};
  
  float radius = 1.0;

  int index = 0;
  int count = 0;
  while (ros::ok())
  {
    count += 1;
    geometry_msgs::PoseStamped pos_msg;
    
    pos_msg.pose.position.x = radius * cos((M_PI * 2) / freq * round_min * count);
    pos_msg.pose.position.y = radius * sin((M_PI * 2) / freq * round_min * count);
    pos_msg.pose.position.z = 1.0;
    
    if (count <= (freq / round_min)) 
    {
      index = count;
      geometry_msgs::PoseStamped temp [index];
      for (int i = 0; i < index - 1; i++)
	temp[i] = path_msg.poses[i];
      temp[index - 1] = pos_msg;
      path_msg.poses = temp;
    }
    else
    {
      geometry_msgs::PoseStamped temp [index];
      for (int i = 0; i < index - 1; i++)
	temp[i] = path_msg.poses[i + 1];
      temp[index - 1] = pos_msg;
      path_msg.poses = temp;
    }

    // ROS_INFO("TEST");

    path_pub.publish(path_msg);
    pos_pub.publish(pos_msg);

    ros::spinOnce();

    loop_rate.sleep();
  }

  return 0;
}
