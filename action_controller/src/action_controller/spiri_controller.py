#! /usr/bin/env python

## @package spiri_controller
# Listens to trajectory and generates the appropriate velocity to move the Quadcopter. 
# @todo This should also convert cmd_vel to pitch
# @author Rohan Bhargava
# @version 1.1.1

import rospy
import trajectory_msgs.msg
import actionlib
from spiri_api import pid_pitch
from collections import deque
from geometry_msgs.msg import Pose, Twist, Vector3
from nav_msgs.msg import Odometry
class spiri_ros_controller(object):
  ## Constructor
  def __init__(self,name):
    self.linear_vel=Vector3()
    rospy.Subscriber("/ground_truth/state",Odometry,self.state_callback)
    rospy.Subscriber("/cmd_vel",Twist,self.vel_callback)


  
  def pid_pitch(self,goal):
    
    #rospy.loginfo("Publishing vel")
    kp=rospy.get_param('kp',2.0)
    ki=rospy.get_param('ki',0.0)
    kd=rospy.get_param('kd',2.0)

    pid_object=pid_pitch.PID([kp,kp,kp],[ki,ki,ki],[kd,kd,kd])
    #pub=rospy.Publisher("cmd_vel",Twist,queue_size=10)
    #vel=Twist()
    pid_object.setPoint(goal)
    current_vel=self.linear_vel
    if abs(current_vel.x-goal.x)<0.1:
      temp=pid_object.update(current_vel)
      print temp[0]
    

  ## Callback for Twist messages
  def vel_callback(self,vel):
    self.pid_pitch(vel.linear)
  
  ## Callback for Odometry messages
  def state_callback(self,data):
    self.linear_vel=data.twist.twist.linear
    
 
    
    





if __name__ == '__main__':
  rospy.init_node('spiri_command_listener',anonymous=True)

  spiri_ros_controller(rospy.get_name())
  rospy.spin()
