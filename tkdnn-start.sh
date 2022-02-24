#!/bin/bash

# Setting camera parameters

v4l2-ctl -d 0 -c focus_auto=0 
v4l2-ctl -d 0 -c focus_absolute=0
v4l2-ctl -d 0 -c led1_mode=0
v4l2-ctl -d 0 -c exposure_auto_priority=1
v4l2-ctl -d 0 -c backlight_compensation=0

cd /usr/local/src/tkDNN/build
./demoGeneral -ini ../config_visagx1-brio.ini 
