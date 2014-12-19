#!/bin/python

""" 

Author: Trevor Gamblin
Date: 2014-12-12

This file contains a basic class for a straight line trajectory flown by Spiri.

"""

import math
import numpy
import rospy
from time import sleep
from Trajectory import *
from geometry_msgs.msg import Twist

class Line(Trajectory):
  
  def __init__(self, length, max_vel = 1, numsteps = 16, pub_rate = 2):
	self.length = length
	self.max_vel = max_vel
	self.numsteps = numsteps
	self.pub_rate = pub_rate
	self.pos = 0
	self.vel = 0
  
  #redefine max_vel. Currently not used
  def set_max_vel(self, v):
	self.max_vel = v
  
  def vet_at_len(self, current_pos):
	self.vel = self.max_vel * (current_pos/self.length)

  #modes for lines are x = 1, y = 2, z = 3
  def ros_publish_line(self, mode = 1):
	
	pub = rospy.Publisher('cmd_vel', Twist, queue_size = 10)
	rospy.init_node('spiri_figures', anonymous = True)
	
	rate = rospy.Rate(self.pub_rate)
	
	vel = Twist()
	output_vel = 0
	
	print self.numsteps
	print self.length
	print self.length/self.numsteps
	for i in numpy.arange(0, self.length+(self.length/self.numsteps), (self.length/self.numsteps)):
	  
	  print "distance travelled is %s" % (i)
	  print "self.vel is %s " % (self.vel) 
	  if i <= 0.5 * self.length:
		self.vel = i * 2 * self.max_vel
	  else:
		self.vel = self.max_vel * (2 / i)
		
	  if (mode == 1):
		vel.linear.x = self.vel
	  elif (mode == 2):
		vel.linear.y = self.vel
	  elif (mode == 3):
		vel.linear.z = self.vel
	  
	  pub.publish(vel)
	  rate.sleep()
	
	print "total distance traveled is %s" % (self.length)
	self.vel = 0
	if (mode == 1):
	  vel.linear.x = self.vel
	elif (mode == 2):
	  vel.linear.y = self.vel
	elif (mode == 3):
	  vel.linear.z = self.vel
	pub.publish(vel)
	rate.sleep()
