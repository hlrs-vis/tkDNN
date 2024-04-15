#!/bin/bash

#compress existing jsons
cd /mnt/sd-card/cameradata/json/
if compgen -G "*.json" > /dev/null; then
    for f in *.json;
    do
       tar -czf $f.gz $f --remove-files
    done
fi

# Setting camera parameters

v4l2-ctl -d 0 -c focus_auto=0 
v4l2-ctl -d 0 -c focus_absolute=0
v4l2-ctl -d 0 -c led1_mode=0
v4l2-ctl -d 0 -c exposure_auto_priority=1
v4l2-ctl -d 0 -c backlight_compensation=0
v4l2-ctl -d 0 -c exposure_auto=1
v4l2-ctl -d 0 -c exposure_absolute=20
v4l2-ctl -d 0 -c brightness=100
v4l2-ctl -d 0 -c gain=128
v4l2-ctl -d 0 -c sharpness=255

# source /etc/profile.d/idsGigETL_64bit.sh
cd /usr/local/src/git/tkDNN/build
./demoGeneral -ini ../config_visagx-brio.ini 
