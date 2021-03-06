#!/usr/bin/env python
from spiri_api import libspiri_api_python as s
import numpy as np
class Position:
  x=0.0
  y=0.0
  z=0.0

class imu:
  x=0.0
  y=0.0
  z=0.0
  w=0.0

class gps:
  latitude=0.0
  longitude=0.0
  altitude=0.0

class State:
  position=Position()
  orientation=imu()
  
class spiri_api_python():
  def __init__(self):
    self.spiri=s.Staterobot()
    
  def get_left_image(self):
    
    image=(np.fromstring(self.spiri.get_left_image(),dtype=np.uint8)).reshape(640,480,3)
    return image
  
  def get_right_image(self):
    
    image=(np.fromstring(self.spiri.get_right_image(),dtype=np.uint8)).reshape(640,480,3)
    return image
  
  def get_bottom_image(self):
    
    image=(np.fromstring(self.spiri.get_bottom_image(),dtype=np.uint8)).reshape(640,480,3)
    return image
  
  
  def get_state(self):
    obj_state=State()
    data=self.spiri.get_state()
    obj_state.position.x=data[0]
    obj_state.position.y=data[1]
    obj_state.position.z=data[2]
    obj_state.orientation.x=data[3]
    obj_state.orientation.y=data[4]
    obj_state.orientation.z=data[5]
    obj_state.orientation.w=data[6]
    return obj_state
  
  def get_imu(self):
    obj_imu=imu()
    data=self.spiri.get_imu()
    obj_imu.x=data[0]
    obj_imu.y=data[1]
    obj_imu.z=data[2]
    obj_imu.w=data[3]
    return obj_imu
  
  
  def get_gps(self):
    obj_gps=gps()
    data=self.spiri.get_gps_data()
    obj_gps.latitude=data[0]
    obj_gps.longitude=data[1]
    obj_gps.altitude=data[2]
    return obj_gps
  
  def get_gps_vel(self):
    obj_gps_vel=Position()
    data=self.spiri.get_gps_vel()
    obj_gps_vel.x=data[0]
    obj_gps_vel.y=data[1]
    obj_gps_vel.z=data[2]
    return obj_gps_vel
  
  
  def get_height_altimeter(self):
    return self.spiri.get_height_altimeter()
  
  def get_height_pressure(self):
    return self.spiri.get_height_pressure()
  
  
  def send_goal(x,y,z,relative=False):
    if relative==False:
      self.spiri.send_goal_relative([x,y,z])
    else:
      self.spiri.send_goal([x,y,z])
 
 def send_vel(x,y,z):
   self.spiri.send_vel([x,y,z])
 