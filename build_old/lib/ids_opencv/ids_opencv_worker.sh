#!/bin/bash

[ "$0" != "$BASH_SOURCE" ] && SOURCED=1 || SOURCED=0
export IDS_PEAK_PATH=$(readlink -f "$(dirname "$BASH_SOURCE")")
export LD_LIBRARY_PATH=/usr/lib/ids_peak-1.3.0/../:$LD_LIBRARY_PATH
export GENICAM_GENTL32_PATH=$GENICAM_GENTL32_PATH:/usr/lib/ids_peak-1.3.0/../ids/cti
export GENICAM_GENTL64_PATH=$GENICAM_GENTL64_PATH:/usr/lib/ids_peak-1.3.0/../ids/cti

[ "$SOURCED" -eq 0 ] && $IDS_PEAK_PATH/ids_opencv_worker $@
