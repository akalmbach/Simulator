#!/usr/bin/env python

## @package spiri_api
# Python API for Spiri
# @author Rohan Bhargava and Arnold Kalmbach
# @version 1.1.3





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

  
## Class defining all the functions to control spiri in Python
class spiri_api_python():
  ## Constuctor
  def __init__(self):
    self.spiri=s.Staterobot()
  
  ## Get image from the left front camera
  # @return Image (640x480)
  def get_left_image(self):
    
    image=(np.fromstring(self.spiri.get_left_image(),dtype=np.uint8)).reshape(640,480,3)
    return image
  
  ## Get image from the right front camera
  # @return Image (640x480)
  def get_right_image(self):
    
    image=(np.fromstring(self.spiri.get_right_image(),dtype=np.uint8)).reshape(640,480,3)
    return image
  
  ## Get image from the bottom camera
  # @return Image (640x480)
  def get_bottom_image(self):
    
    image=(np.fromstring(self.spiri.get_bottom_image(),dtype=np.uint8)).reshape(640,480,3)
    return image
  
  ## Get the state
  # Position and orientation
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
  ## Get the orientation in quaternion from IMU
  # @return  Orientation of (x,y,z,w) of Spiri
  def get_imu(self):
    obj_imu=imu()
    data=self.spiri.get_imu()
    obj_imu.x=data[0]
    obj_imu.y=data[1]
    obj_imu.z=data[2]
    obj_imu.w=data[3]
    return obj_imu
  
  ## Get the gps data
  # @return Latitude, longitude and altitude
  def get_gps(self):
    obj_gps=gps()
    data=self.spiri.get_gps_data()
    obj_gps.latitude=data[0]
    obj_gps.longitude=data[1]
    obj_gps.altitude=data[2]
    return obj_gps
  
  ## Get the velocity reported by GPS
  # @return Velocity (x,y,z) of Spiri
  def get_gps_vel(self):
    obj_gps_vel=Position()
    data=self.spiri.get_gps_vel()
    obj_gps_vel.x=data[0]
    obj_gps_vel.y=data[1]
    obj_gps_vel.z=data[2]
    return obj_gps_vel
  
  ## Get the height in metres from altimeter
  # Altitude of Spiri
  def get_height_altimeter(self):
    return self.spiri.get_height_altimeter()
  
  ## Get the height in metres from pressure sensor
  # Altitude of Spiri
  def get_height_pressure(self):
    return self.spiri.get_height_pressure()
  
  ## Send goal to Spiri
  # @param x coordinate in x direction
  # @param y coordinate in y direction
  # @param z coordinate in z direction
  # @param relative If set the goal is calculated with respect to the start position otherwise coordinates are with respect to the world
  def send_goal(self,x,y,z,relative=False):
    if relative==True:
      self.spiri.send_goal_relative([x,y,z])
    else:
      self.spiri.send_goal([x,y,z])
  ## Send velocity to Spiri
  # @param x velocity in x direction
  # @param y velocity in y direction
  # @param z velocity in z direction
  
  def send_vel(self,x,y,z):
   self.spiri.send_vel([x,y,z])
  ## If the goal has been reached or not
  # @return True if goal has been reached otherwise False
  def wait_goal(self):
    return self.spiri.wait_goal()

  
  def stop_traj(self):
    return self.spiri.stop_traj()


  def takeoff(self):
    return self.spiri.takeoff()

  def land(self):
    return self.spiri.land()

    
 
