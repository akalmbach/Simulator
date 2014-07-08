#Simulator


This branch contains the alpha release of the simulator.

It contains the following functionalities-:

- Spawn Spiri in Gazebo with a choice of a world

- Publish sensor data on individual topics.

- Control Spiri in the simulator by a joystick.

The code has been tested on ROS Hydro and Ubuntu 12.04

## Installation instructions



- Install ROS

http://wiki.ros.org/hydro/Installation/Ubuntu

Install the ros-desktop-full package as it contains Gazebo and other programs required for the simulator.

- Install Gazebo sensor plugins

sudo apt-get install ros-hydro-hector-quadrotor

- Install the Controllers

sudo apt-get install ros-hydro-ros-controllers

- Install joystick drivers

sudo apt-get install ros-hydro-joy

- Create a ros workspace

http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment

After this step there should be a folder called as catkin_ws

- Get the Simulator

```bash
cd catkin_ws/src

git clone https://github.com/Pleiades-Spiri/Simulator.git -b alpha-release

cd ..

catkin_make
```

These commands will build the code required for the Simulator.

Make sure these commands are in your bashrc
```bash

nano ~/.bashrc

source ~/catkin_ws/devel/setup.bash

source /opt/ros/hydro/setup.bash
```

## Commands


To launch Spiri in an empty world

```bash
roslaunch spiri_description spiri_empty_world.launch
```

To launch Spiri in Citadel Hill

```bash
roslaunch spiri_description spiri_citadel.launch
```

Control Spiri with a joystick

Xbox controller 

```bash
roslaunch spiri_teleop xbox_controller.launch
```

Logitech Gamepad

```bash
roslaunch spiri_teleop logitech_gamepad.launch
```  


Script to process the Depth Map

```bash
ROS_NAMESPACE=stereo rosrun stereo_image_proc stereo_image_proc _approximate_sync:=True _queue_size:=10
```

View the Depth Map

```bash
rosrun image_view stereo_view stereo:=/stereo image:=image_rect_color _approximate_sync:=True _queue_size:=10
```


## To visualize the robot and point cloud data in rviz

```bash
roscd spiri_description

rosrun rviz rviz -d spiri_pointcloud.rviz
```

## Nodes

### gazebo
Launches spiri in a world with all its sensors

#### Subscribed topics 

- /cmd_vel
- /wind
#### Published Topics

- /altimeter
- /fix
- /fix/pose
- /ground_truth/state
- /raw_imu
- /pressure_height
- /magnetic
- /stereo/left/image_raw
- /stereo/right/image_raw


## Troubleshooting

### Error [ConnectionManager.cc:116] Failed to connect to master in 30 seconds

This is a known bug in Gazebo

https://bitbucket.org/osrf/gazebo/issue/732/gazebo-never-starts-when-loading-models

To fix this do the following steps-:

```bash
cd catkin_ws/src

hg clone https://bitbucket.org/osrf/gazebo_models

Add this command to your bashrc

export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/catkin_ws/src/gazebo_models
```

Restart your terminal and it should fix the error
