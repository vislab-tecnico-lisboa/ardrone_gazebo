#include <iostream>

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>

class RobotDriver
{
private:
  //! The node handle we'll be using
  ros::NodeHandle nh_;
  //! We will be publishing to the "/base_controller/command" topic to issue commands
  ros::Publisher cmd_vel_pub_;

public:
  //! ROS node initialization
  RobotDriver(ros::NodeHandle &nh)
  {
    nh_ = nh;
    //set up the publisher for the cmd_vel topic
    cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/vizzy/cmd_vel", 1);
  }

  //! Loop forever while sending drive commands based on keyboard input
  bool driveKeyboard()
  {
    std::cout << "Type a command and then press enter.  "
      "Use 'w' to move forward, 'a' to turn left, "
      "'d' to turn right, 's' to brake.\n"
      "'x' to move backward, '.' to exit.\n";

    //we will be sending commands of type "twist"
    geometry_msgs::Twist base_cmd;

    char cmd[50];
    while(nh_.ok()){

      std::cin.getline(cmd, 50);
      if(cmd[0]!='w' && cmd[0]!='a' && cmd[0]!='d' && cmd[0]!='s' && cmd[0]!='x' && cmd[0]!='.')
      {
        std::cout << "unknown command:" << cmd << "\n";
        continue;
      }

      base_cmd.linear.x = base_cmd.linear.y = base_cmd.angular.z = 0;   
      //move forward
      if(cmd[0]=='w'){
        base_cmd.linear.x = 0.50;
      } 
      //turn left (yaw) and drive forward at the same time
      else if(cmd[0]=='a'){
        base_cmd.angular.z = 0.75;
        base_cmd.linear.x = 0.25;
      } 
      //turn right (yaw) and drive forward at the same time
      else if(cmd[0]=='d'){
        base_cmd.angular.z = -0.75;
        base_cmd.linear.x = 0.25;
      } 
      //brake
      else if(cmd[0]=='s'){
        base_cmd.linear.x = 0.0;
      }
      //brake
      else if(cmd[0]=='x'){
        base_cmd.linear.x = -0.50;
      } 
      //quit
      else if(cmd[0]=='.'){
        break;
      }

      //publish the assembled command
      cmd_vel_pub_.publish(base_cmd);
    }
    return true;
  }

};

int main(int argc, char** argv)
{
  //init the ROS node
  ros::init(argc, argv, "robot_driver");
  ros::NodeHandle nh;

  RobotDriver driver(nh);
  driver.driveKeyboard();
}