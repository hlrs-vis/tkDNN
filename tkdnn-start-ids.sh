#!/bin/bash

# sourcing ids-components


source /etc/profile.d/idsGigETL_64bit.sh
cd /usr/local/src/git/tkDNN/build
./demoGeneral -ini ../config_visagx-ids.ini 
