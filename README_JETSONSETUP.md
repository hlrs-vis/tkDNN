# Setting up the Jetson for NVMe Try 1
## Host Computer
x84_64 Desktop needed, cant flash from another Jetson.
Make sure [Nvidia SDK Manager](https://developer.nvidia.com/sdk-manager) is installed.
Connect Target Jetson via USB to Host. (USB-C Port on Jetson Side)
Open SDK Manager.

### SDK Manager
#### Step 1
Put Jetson into Recovery Mode.
If it is turned off:
1. Press and hold down the Force Recovery button.
2. Press and hold down the Power button.
3. Release both buttons

If it is already on:
1. Press and hold down the Force Recovery button.
2. Press the RESET button

SDK Manager should automatically detect the Device.

#### Step 2
Choose what to install.
Runtime components can be installed later, seperately from Image.

#### Step 3
Image will now be downloaded.
Complete Setup about OEM Config and Storage Device.


After installation Jetson didnt boot.

# Setting up Jetson NVMe Try 2

Prerequisite:
Already configured and bootable nvme device attached to jetson.

Start at Host Computer in Try 1, but only install the Cuda runtime, dont flash the jetson.

# Setting up Jetson NVMe Try 2 Prerequisite, configure NVMe device
## Prerequisite
Bootable eMMC (internal) storage Jetson.
Host Computer with [Nvidia SDK Manager](https://developer.nvidia.com/sdk-manager).
## Setting up Jetson NVMe from scratch.
Put NVMe in Host Computer.
[NVIDIA DOC](https://docs.nvidia.com/jetson/archives/r35.4.1/DeveloperGuide/text/SD/FlashingSupport.html#flashing-to-an-nvme-drive) see Documentation for questions.
Figure out NVMe drive's name
```
$ lsblk -d -p | grep nvme | cut -d\  -f 1
```
Run command (see Documentation for the variables):
```
$ sudo <env-var> ./tools/kernel_flash/l4t_initrd_flash.sh [ -S <rootfssize> ] -c <config> --external-device nvme0n1p1 --direct <nvmeXn1> <board> external
```
For Jetson AGX Xavier use:
```
$ sudo BOARDID=2888 BOARDSKU=0003 FAB=TS4 ./tools/kernel_flash/l4t_initrd_flash.sh -c tools/kernel_flash/flash_l4t_external.xml --external-device nvme0n1p1 --direct nvme0n1p1 jetson-agx-xavier-devkit external
```

Put NVMe back into Jetson.
Boot Jetson and select NVMe as booting option.
Specify Partition Size and do initial setup, then resume at Setting up Jetson NVMe Try 2


