#!/bin/python

""" 

Author: Trevor Gamblin
Date: 2014-12-11

This file is a test bed for flying circular trajectories with Spiri in Gazebo

"""

from Trajectory import *
from Circular import *
import rospy
import math

def main():
  
  helix1 = Helix(math.pi*2, 1, 0.0, 16, 5)
  #helix1.ros_publish_helix(mode = 1, vel_multiplier = 1)
  #helix1.stop_all_vel()
  #helix1.ros_publish_helix(mode = 2, vel_multiplier = 1)
  #helix1.stop_all_vel()
  #helix1.ros_publish_helix(mode = 3, vel_multiplier = 1)
  #helix1.stop_all_vel()
  #helix1.ros_publish_helix(mode = 4, vel_multiplier = 1)
  #helix1.stop_all_vel()
  helix1.ros_publish_helix(mode = 5, vel_multiplier = 1)
  helix1.stop_all_vel()
  helix1.ros_publish_helix(mode = 6, vel_multiplier = 1)
  helix1.stop_all_vel()
  helix1.ros_publish_helix(mode = 7, vel_multiplier = 1)
  helix1.stop_all_vel()
  helix1.ros_publish_helix(mode = 8, vel_multiplier = 1)
  helix1.stop_all_vel()
  
if __name__ == "__main__":
  try:
    main()
  except rospy.ROSInterruptException:
    pass
