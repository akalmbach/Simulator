#!/bin/python

""" 

Author: Trevor Gamblin
Date: 2014-12-08

This file contains the fundamental class for all trajectory types

"""

import math

class Trajectory(object):

  def __init__(self, arclen):
    self.arclen = arclen
    self.vel_x = 0
    self.vel_y = 0
    self.vel_z = 0
    
  def __del__(self):
    class_name = self.__class__.__name__
    print class_name, "destroyed"
  
  def set_vel_x(self, in_vel):
    self.vel_x = in_vel
  
  def set_vel_y(self, in_vel):
    self.vel_y = in_vel
    
  def set_vel_z(self, in_vel):
    self.vel_z = in_vel
    
  def get_vel_x(self):
    return self.vel_x
  
  def get_vel_y(self):
    return self.vel_y
    
  def get_vel_z(self):
    return self.vel_z
    
  def vel_z(self, c):
    self.vel_z = c
    
  #stop_all_vel sets all linear velocities to 0
    
  def stop_all_vel(self):
    self.vel_x = 0.0
    self.vel_y = 0.0
    self.vel_z = 0.0

#these two functions are used to calculate velocities on a circular curve

  def sin_vel(self, r, t):
	return r * math.sin(t)
    
  def cos_vel(self, r, t):
	return r * math.cos(t)
