#!/bin/python

""" 

Author: Trevor Gamblin
Date: 2014-12-08

This file contains a basic class for a helical trajectory. Setting the z_rate = 0 results in a circle

"""

import math
import numpy
import rospy
from time import sleep
from Trajectory import *
from geometry_msgs.msg import Twist

class Helix(Trajectory):
  def __init__(self, arclen, radius, z_rate, numsteps, pub_rate):
    self.arclen = arclen
    self.radius = radius
    self.z_rate = z_rate
    self.numsteps = numsteps
    self.pub_rate = pub_rate
    self.vel_x = 0
    self.vel_y = 0
    self.vel_z = 0
    self.s_curr = 0
    self.t_curr = 0
    
      
#s_func_t and t_func_s set Helix current arc length and time values
    
  def s_func_t(self, a, c, t):
    a_term = a * a
    c_term = a * a
    self.s_curr = math.sqrt(a_term + c_term) * t
    
  def t_func_s(self, a, c, s):
    a_term = a * a
    c_term = c * c
    self.t_curr = s/math.sqrt(a_term + c_term)
  
  #below are dictionary functions and a dictionary for the helix modes.
  #modes 1,2,3,4 are ccw while 5,6,7,8 are cw. Mode 0 is undefined. 1, 2, 3, 4 specify which Cartesian quadrant 
  #the ccw helix will be performed in; 5, 6, 7, 8 specify which (n-4) quadrant the cw motion will occur in
  #if 1 <= flight mode <= 4 then the pattern will be ccw. If 5 <= flight_mode <= 8, it will be cw
  
  def helix_mode_1(self, r, t):
	self.vel_x = self.cos_vel(r, t)
	self.vel_y = self.sin_vel(r, t)
  
  def helix_mode_2(self, r, t):
	self.vel_x = self.sin_vel(r, t) * (-1)
	self.vel_y = self.cos_vel(r, t)
  
  def helix_mode_3(self, r, t):
	self.vel_x = self.cos_vel(r, t) * (-1)
	self.vel_y = self.sin_vel(r, t) * (-1)
  
  def helix_mode_4(self, r, t):
	self.vel_x = self.sin_vel(r, t)
	self.vel_y = self.cos_vel(r, t) * (-1)
  
  def helix_mode_5(self, r, t):
	self.vel_x = self.cos_vel(r * (-1), t)
	self.vel_y = self.sin_vel(r * (-1), t)
  
  def helix_mode_6(self, r, t):
	self.vel_x = self.sin_vel(r * (-1), t) * (-1)
	self.vel_y = self.cos_vel(r * (-1), t)
  
  def helix_mode_7(self, r, t):
	self.vel_x = self.cos_vel(r * (-1), t) * (-1)
	self.vel_y = self.sin_vel(r * (-1), t) * (-1)
  
  def helix_mode_8(self, r, t):
	self.vel_x = self.sin_vel(r * (-1), t)
	self.vel_y = self.cos_vel(r * (-1), t) * (-1)
    
#ros_publish_helix publishes necessary data to ROS

  def ros_publish_helix(self, mode = 1, vel_multiplier = 1):
    
	pub = rospy.Publisher('cmd_vel', Twist, queue_size = 10)
	rospy.init_node('spiri_figures', anonymous = True)
      
	rate = rospy.Rate(self.pub_rate)
      
	#goal_radius = self.radius
	#goal_arclen = self.arclen
	#steps = self.numsteps
	#z_rate = self.z_rate
	
	flight_mode = mode
	vel_mult = vel_multiplier
	vel=Twist()
	#i = self.t_curr
	#j = self.s_curr
	i = 0
      
	while i < self.arclen:
	  #print "goal_arclen is %s" % (goal_arclen)
	  arc_length = self.s_curr
      
	  if (1 <= flight_mode <= 4):
		for i in numpy.arange(0, self.arclen+(math.pi/self.numsteps), math.pi/self.numsteps):
		  self.t_func_s(self.radius, self.z_rate, i)
		  print "i is %s" % (i)
		  print "t_curr is %s" % (self.t_curr)
		  print "X vel: %s, Y vel: %s, arc length: %s" % (self.vel_x, self.vel_y, self.s_curr)
		  if flight_mode == 1:
			self.helix_mode_1(self.radius, self.t_curr)
		  elif flight_mode == 2:
			self.helix_mode_2(self.radius, self.t_curr)
		  elif flight_mode == 3:
			self.helix_mode_3(self.radius, self.t_curr)
		  elif flight_mode == 4:
			self.helix_mode_4(self.radius, self.t_curr)
			
		  self.vel_z = self.z_rate
		  vel.linear.x = self.vel_x * vel_mult
		  vel.linear.y = self.vel_y * vel_mult
		  vel.linear.z = self.vel_z * vel_mult
      
		  pub.publish(vel)
		  rate.sleep()
		if i >= self.arclen:
		  self.stop_all_vel()
		  vel.linear.x = self.vel_x
		  vel.linear.y = self.vel_y
		  vel.linear.z = self.vel_z
		  print "vel.linear.x is %s, self.vel_x is %s " % (vel.linear.x, self.vel_x)
		  print "vel.linear.y is %s, self.vel_y is %s " % (vel.linear.y, self.vel_y)
		  print "vel.linear.z is %s, self.vel_z is %s " % (vel.linear.z, self.vel_z)
		  
		  pub.publish(vel)
		  rate.sleep()
		  
	  elif (5 <= flight_mode <= 8):
		for i in numpy.arange(0, self.arclen+(math.pi/self.numsteps), math.pi/self.numsteps):
		  self.t_func_s(self.radius, self.z_rate, i)
		  print "i is %s" % (i)
		  print "t_curr is %s" % (self.t_curr)
		  print "X vel: %s, Y vel: %s, arc length: %s" % (self.vel_x, self.vel_y, self.s_curr)
		  if flight_mode == 5:
			self.helix_mode_5(self.radius, self.t_curr * (-1))
		  elif flight_mode == 6:
			self.helix_mode_6(self.radius, self.t_curr * (-1))
		  elif flight_mode == 7:
			self.helix_mode_7(self.radius, self.t_curr * (-1))
		  elif flight_mode == 8:
			self.helix_mode_8(self.radius, self.t_curr * (-1))
		  
		  self.vel_z = self.z_rate
		  vel.linear.x = self.vel_x * vel_mult
		  vel.linear.y = self.vel_y * vel_mult
		  vel.linear.z = self.vel_z * vel_mult
		  print "vel.linear.x is %s, self.vel_x is %s " % (vel.linear.x, self.vel_x)
		  print "vel.linear.y is %s, self.vel_y is %s " % (vel.linear.y, self.vel_y)
		  print "vel.linear.z is %s, self.vel_z is %s " % (vel.linear.z, self.vel_z)
      
		  pub.publish(vel)
		  rate.sleep()
		if i >= self.arclen:
		  self.stop_all_vel()
		  vel.linear.x = self.vel_x
		  vel.linear.y = self.vel_y
		  vel.linear.z = self.vel_z
		  pub.publish(vel)
		  rate.sleep()
		  
