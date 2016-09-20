# ardrone_gazebo
Gazebo simulator test environment for the Parrot ArDrone. The world is a simulated version of the ISR 7th floor. The ArDrone model is based on the implementation of a gazebo simulator for the Ardrone 2.0 written by Hongrong Huang and Juergen Sturm of the Computer Vision Group at the Technical University of Munich (http://wiki.ros.org/tum_simulator). 

Installation instructions:

1 - install ros full desktop following the installation instructions on the official ros website: www.ros.org (tested on indigo, jade and kinetic)

2 - install the ardrone_autonomy package. If you are on Ubuntu simply write on your console:
$ sudo apt-get install ros-<your-ros-distribution>-ardrone_autonomy

3 - if you are using ros indigo install gazebo5, 6 or 7 from the osrfoundation repository. look at this page for more details: http://gazebosim.org/tutorials?tut=ros_wrapper_versions

4 - if you don't have it already, create a catkin workspace folder (for more informations look at this link: http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment):
$ mkdir catkin_ws
Create a folder named src inside it:
$ cd catkin_ws
$ mkdir src
Run catkin_init_workspace inside the src directory
$ cd src
$ catkin_init_workspace
Now source your new setup.bash file inside your .bashrc
$ echo "source <your_catkin_ws_directory>/devel/setup.bash" >> ~/.bashrc

5 - clone this git repository inside your catkin workspace src directory
$ cd <your_catkin_ws_directory>/src
$ git clone 
