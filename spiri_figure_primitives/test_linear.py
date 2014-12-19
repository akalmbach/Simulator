#!/bin/python

""" 

Author: Trevor Gamblin
Date: 2014-12-17

This file is a test bed for flying linear trajectories with Spiri in Gazebo

"""

from Trajectory import *
from Linear import *
import rospy
import math

def main():
  
  linear1 = Line(length = 4.0, max_vel = 1.0, numsteps = 16, pub_rate = 2)
  linear1.ros_publish_line(mode = 3)
  linear2 = Line(length = 4.0, max_vel = -1, numsteps = 16, pub_rate = 2)
  linear2.ros_publish_line(mode = 3)
  linear3 = Line(length = 4.0, max_vel = 1.0, numsteps = 8, pub_rate = 2)
  linear3.ros_publish_line(mode = 3)
  linear4 = Line(length = 4.0, max_vel = -1.0, numsteps = 8, pub_rate = 2)
  linear4.ros_publish_line(mode = 3)
  linear5 = Line(length = 4.0, max_vel = 1.0, numsteps = 8, pub_rate = 4)
  linear5.ros_publish_line(mode = 3)
  linear6 = Line(length = 4.0, max_vel = -1.0, numsteps = 8, pub_rate = 4)
  linear6.ros_publish_line(mode = 3)
 
if __name__ == "__main__":
  try:
    main()
  except rospy.ROSInterruptException:
    pass
