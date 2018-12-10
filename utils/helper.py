#!/usr/bin/env python
"""
Helper funtions for steering angle prediction model
"""

import  numpy as np
import  tensorflow as tf
from    utils.config import *

class PreProcessor():
    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("pre-processor"):
            self.inputImg = tf.placeholder(shape=[None, 1, config.imgCh, config.imgRow, config.imgCol], dtype=tf.float32)
            self.steering = tf.placeholder(shape=[None, 1, 1], dtype=tf.float32)
            self.speed    = tf.placeholder(shape=[None, 1, 1], dtype=tf.float32)
        
            # Color conversion (rgb to yuv) and normalization
        self.outImg   = tf.squeeze(tf.transpose(self.inputImg, perm=[0, 3, 4, 2, 1]), [4]) # [None, config.imgRow, config.imgCol, config.imgCh]
        self.outImg   = tf.image.resize_nearest_neighbor(self.outImg, [int(config.imgRow/config.resizeFactor), int(config.imgCol/config.resizeFactor)])
        self.outImg   = self.outImg/255.
        self.outImg   = tf.image.rgb_to_hsv(self.outImg)

        # Convert Steering angle to Curvature
        self.steering_  = tf.squeeze(self.steering, [1]) # /10.
        self.speed_     = tf.squeeze(self.speed, [1])
        self.calc_curvature()
        self.curvature  = self.curvature_ * 100000

    # brought from comma.ai
    def calc_curvature(self, angle_offset=0):
        # slip is the relative motion between a tire and the road surface it is moving on
        # Steering ratio refers to the ratio between the turn of the steering wheel (in degrees) or handlebars and the turn of the wheels (in degrees)
        # wheelbase is the distance between the centers of the front and rear wheels
        deg_to_rad  = np.pi/180.
        slip_fator  = 0.0014 # slip factor obtained from real data
        steer_ratio = 15.3  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
        wheel_base  = 2.67   # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

        angle_steers_rad   = (self.steering_/10. - angle_offset) * deg_to_rad
        self.curvature_    = angle_steers_rad/(steer_ratio * wheel_base * (1. + slip_fator * self.speed_**2))

    # process pre-porcessing
    def process(self, sess, inImg, steering, speed):
        feed = {self.inputImg: inImg, self.steering: steering, self.speed: speed}
        return sess.run([self.outImg, self.curvature, self.steering_], feed)

class PreProcessor_VA():
    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("pre-processor"):
            self.feats    = tf.placeholder(shape=[config.epoch, config.timelen, 64, 10, 20], dtype=tf.float32)
            self.steering = tf.placeholder(shape=[config.epoch, config.timelen, 1], dtype=tf.float32)
            self.speed    = tf.placeholder(shape=[config.epoch, config.timelen, 1], dtype=tf.float32)

            # reshape
            self.outFeats   = tf.reshape(   self.feats, [config.epoch, config.timelen, 64, 200] )
            self.outFeats   = tf.reshape(   self.outFeats, 	[-1, 64, 200] )
            self.outFeats   = tf.transpose( self.outFeats, 	[0,2,1] ) # [batchsize,20,200,64]

            # Convert Steering angle to Curvature
            self.calc_curvature()
            self.curvature  = tf.reshape( self.curvature_, [-1, 1])
            self.curvature  = self.curvature * 100000

            self.steering_  = tf.reshape( self.steering, [-1, 1] )

    # brought from comma.ai
    def calc_curvature(self, angle_offset=0):
        # slip is the relative motion between a tire and the road surface it is moving on
        # Steering ratio refers to the ratio between the turn of the steering wheel (in degrees) or handlebars and the turn of the wheels (in degrees)
        # wheelbase is the distance between the centers of the front and rear wheels
        deg_to_rad  = np.pi/180.
        slip_fator  = 0.0014 # slip factor obtained from real data
        steer_ratio = 15.3  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
        wheel_base  = 2.67   # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

        angle_steers_rad    = (self.steering/10. - angle_offset) * deg_to_rad
        self.curvature_     = angle_steers_rad/(steer_ratio * wheel_base * (1. + slip_fator * self.speed**2))

    # process pre-porcessing
    def process(self, sess, feats, steering, speed):
        feed = {self.feats: feats, self.steering: steering, self.speed: speed}
        return sess.run([self.outFeats, self.curvature, self.steering_], feed)





