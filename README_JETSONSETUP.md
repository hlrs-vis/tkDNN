#Setting up the Jetson Try 1
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

