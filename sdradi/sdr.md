
# Implementing Open Hardware Basestation Systems for future 5G/6G

## Introduction

Significant development efforts were dedicated to implementing an open hardware system designed to serve as the future software-defined 5G/6G base station prototype. This system is crucial for future 5G/6G wireless communication networks, enabling seamless connectivity, efficient data transfer, as well as advanced sensing and intelligent processing capabilities. It also allows for full software control and development of communication and sensing functionalities. Specifically, we developed two versions of the system in parallel, each targeting different deployment scenarios.

## Prototype Basestation Systems

### 1. Wideband Base Station Device

Our wideband base station device serves as the cornerstone of our base station system. It supports a wideband Orthogonal Frequency Division Multiplexing (OFDM) communication scheme with an impressive bandwidth of 200 MHz, adjustable via software from 75 MHz to 6 GHz. The software-defined radio transceiver features dual transmitters and dual receivers connected to MIMO antennas with integrated low-noise amplifiers (LNA). This configuration enables high-speed data transmission, making it suitable for research into next-generation networks like 5G and 6G. 

Beyond communication, the device incorporates radar sensing capabilities with a wide bandwidth of 400 MHz. This feature supports advanced applications such as target detection, environmental monitoring, and security surveillance. We utilize an additional observation receiver bandwidth of 450 MHz as the radar receiver, enabling a single device to support both communication and radar sensing. In addition to these fundamental radio features, our wideband base station device is equipped with an **Intel Arria10 FPGA** as the baseband processing unit and an on-board **NVIDIA Jetson Orin GPU** with 40 TOPS processing power as the main processor. 

Additional components in our Wideband Base Station Device include: onboard cameras, a Robosense 16-channel LiDAR, a solar panel and battery system, a PoE++ standard switch, and a power supply.

### 2. Cost-Effective Mobile Node

Complementing the wideband base station, our cost-effective mobile node offers an alternative solution for specific deployment scenarios. The mobile node provides a narrower software-defined communication bandwidth of 56 MHz. While not as extensive as the wideband base station, this bandwidth is sufficient for high-speed communication with a small number of nearby devices within smaller coverage areas. 

Due to the limited 56 MHz bandwidth, which is insufficient for radar sensing, we integrated an external frequency modulation chip outside the transceiver. This chip enables frequency sweeping of 500 MHz, granting the mobile node a frequency-modulated continuous wave (FMCW) radar sensing capability with a bandwidth of 500 MHz. This feature allows for precise distance measurements, making the mobile node suitable for applications such as target detection, positioning, and obstacle detection.


# SDR Devices

## USB connect to ADALM-PLUTO

ADALM-PLUTO is based on Analog Devices AD9363--Highly Integrated RF Agile Transceiver and Xilinx® Zynq Z-7010 FPGA
  * ADALM-PLUTO Overview: https://wiki.analog.com/university/tools/pluto
  * Website: https://www.analog.com/en/design-center/evaluation-hardware-and-software/evaluation-boards-kits/adalm-pluto.html#eb-overview
  * RF coverage from 325 MHz to 3.8 GHz
  * Up to 20 MHz of instantaneous bandwidth
  * up to 61.44 Mega Samples per Second (MSPS)
  * ADI Book Software-Defined Radio for Engineers, 2018: https://www.analog.com/en/education/education-library/software-defined-radio-for-engineers.html
  * Analog Devices Board Support Packages Toolbox For MATLAB and Simulink: https://wiki.analog.com/resources/eval/user-guides/matlab_bsp

ADALM-PLUTO hardware: https://wiki.analog.com/university/tools/pluto/hacking/hardware
  * The PlutoSDR includes a button (S1 on the PCB), and two USB connectors. The first USB connector (the middle one) is the USB OTG connector (can be the USB HOST connector (cabled to a USB peripheral), or the USB peripheral connector (cabled to a USB Host)). The second USB connector (the one on the side) is for power only when running in Host mode.
  * New Rev D features: addition of internal U.FL connectors for: second receive channel, second transmit channel, Clock input, Clock output (only a copy of Clock input, not functional for the internal clock); USB UART; breakout pins for I2C and SPI; 3.3V GPO levels
  * Schematic: https://wiki.analog.com/_media/university/tools/pluto/hacking/plutosdr_schematic_revd_0.1.pdf
  * BOM: https://wiki.analog.com/_media/university/tools/pluto/hacking/plutosdr_bom_revd.xlsx
  * Allegro project: https://wiki.analog.com/_media/university/tools/pluto/hacking/plutosdr_brd_revd.zip
  * Cadence project: https://wiki.analog.com/_media/university/tools/pluto/hacking/plutosdr_cadence_revd.zip



ADALM-PLUTO for End Users: https://wiki.analog.com/university/tools/pluto/users
  * libiio USB device for communicating to the RF device
  * enumerate with the 192.168.2.1 IP address by default.
  * provides access to the Linux console on the Pluto device via USB Communication Device Class Abstract Control Model (USB CDC ACM) specification
  * Windows driver: https://wiki.analog.com/university/tools/pluto/drivers/windows
  * Linux driver: https://wiki.analog.com/university/tools/pluto/drivers/linux
  * MATLAB: https://www.mathworks.com/hardware-support/adalm-pluto-radio.html
    * Install Support Package for Analog Devices ADALM-PLUTO Radio: https://www.mathworks.com/help/supportpkg/plutoradio/ug/install-support-package-for-pluto-radio.html
    * Setup: https://www.mathworks.com/help/supportpkg/plutoradio/ug/guided-host-radio-hardware-setup.html
    * Manual Setup: https://www.mathworks.com/help/supportpkg/plutoradio/ug/manual-host-radio-hardware-setup.html
  * PlutoSDR (using python bindings to libiio): https://github.com/radiosd/PlutoSdr
  * pyadi-iio: https://wiki.analog.com/resources/tools-software/linux-software/pyadi-iio, https://analogdevicesinc.github.io/pyadi-iio/
  * GNU Radio and IIO Devices: gr-iio: https://wiki.analog.com/resources/tools-software/linux-software/gnuradio
  * Accessing Pluto's FPGA Over JTAG: https://wiki.analog.com/university/tools/pluto/devs/fpga
  * HDL code: https://github.com/analogdevicesinc/hdl/tree/master/projects/pluto

### Getting started for Windows
Plug the ADALM-PLUTO to the USB port of the HOST PC. Download and install the Windows driver from [link](https://wiki.analog.com/university/tools/pluto/drivers/windows), Once the drivers are installed, and the device (Pluto or M2k) is plugged in, the following subsystems should be ready to use:
  * USB Ethernet/RNDIS Gadget. It provides a virtual Ethernet link to most versions of the Windows, Linux and OS X operating systems. The IP address of the PLUTO device is 192.168.2.1.
  * To a host, the usb device acts as an external hard drive. There will be one drive for the pluto device. Open `info.html` inside the drive to see the device information. Under “Build Settings”. By default, it is username: root ; password is analog.
  * Serial Console (115200-8N1), in this case COM15, but it will be different on your PC. The terminal settings are 115200 baud, 8 bits, no parity, 1 stop bit. This is referred to as 115200-8N1. The default username is root, and the default root password is analog.
  * IIO USBD
  * Install [libiio](https://github.com/analogdevicesinc/libiio) under conda/python environment



```bash
(mycondapy310) PS D:\Developer\radarsensing> iio_info -s
Unable to create Local IIO context : Function not implemented (40)
Available contexts:
        0: 0456:b673 (Analog Devices Inc. PlutoSDR (ADALM-PLUTO)), serial=1044739a470b00060c00240091b07e0294 [usb:2.11.5]
(mycondapy310) PS D:\Developer\radarsensing> iio_attr -a -C fw_version
Using auto-detected IIO context at URI "usb:2.11.5"
fw_version: v0.37
```

Connect the device via ssh
```bash
ssh-keygen -R 192.168.2.16 #remove the entry from known_hosts
ssh root@192.168.2.16 #get the ip from the `info.html` page, password: analog
```

Pluto firmware update: https://wiki.analog.com/university/tools/pluto/users/firmware
  * Download the firmware (`plutosdr-fw-v0.38`), copy the entire unzipped firmware files to the Mass Storage device
  * Eject (don't unplug) the mass storage device, this will cause LED1 to blink rapidly. This means programming is taking place. Do not remove power (or USB) while the device is blinking rapidly. It does take approximately 4 minutes to properly program the device. Once the device is done programming, it will re-appear as a mass storage device. Now you can unplug it, and use it as normal.

```bash
(mycondapy310) PS D:\Developer\radarsensing> iio_attr -a -C fw_version 
Using auto-detected IIO context at URI "usb:2.12.5"
fw_version: v0.38
(mycondapy310) PS D:\Developer\radarsensing> iio_attr -a -C
Using auto-detected IIO context at URI "usb:2.12.5"
IIO context with 15 attributes:
hw_model: Analog Devices PlutoSDR Rev.C (Z7010-AD9363A)
hw_model_variant: 1
hw_serial: 1044739a470b00060c00240091b07e0294
fw_version: v0.38
ad9361-phy,xo_correction: 39999971
ad9361-phy,model: ad9363a
local,kernel: 5.15.0-175882-ge14e351533f9
uri: usb:2.12.5
usb,idVendor: 0456
usb,idProduct: b673
usb,release: 2.0
usb,vendor: Analog Devices Inc.
usb,product: PlutoSDR (ADALM-PLUTO)
usb,serial: 1044739a470b00060c00240091b07e0294
usb,libusb: 1.0.26.11724
(mycondapy310) PS D:\Developer\radarsensing> iio_info -n pluto.local
(mycondapy310) PS D:\Developer\radarsensing> iio_info -n pluto
(mycondapy310) PS D:\Developer\radarsensing> iio_info -u ip:192.168.2.1
```

[Customizing the Pluto configuration](https://wiki.analog.com/university/tools/pluto/users/customizing)
  * IETF reserve the IPv4 address range the 192.168.*.* (and others) for private networks. Analog Devices picked the 192.168.2.* subnet for it's private network for host to PlutoSDR devices
  * When using multiple PlutoSDR devices on the same host, there are a few options: usb mode via libiio, no changes are required, and things will work out of the box; network mode, where changes to the network settings are required. In order to use multiple devices, you must change their IP address. This is managed by updating the `config.txt` file on the PlutoSDR mass storage device

```
(mycondapy310) PS D:\Developer\radarsensing> ssh root@192.168.2.1 #password: analog
# fw_printenv attr_name
## Error: "attr_name" not defined
# fw_printenv attr_val
## Error: "attr_val" not defined
# fw_setenv attr_name compatible
# fw_setenv attr_val ad9364
# fw_setenv compatible ad9364
# reboot
```

Update the Pluto configuration to enable the AD9361's second channel, i.e., enable `2r2t`
```
(mycondapy310) PS D:\Developer\radarsensing> ssh-keygen -R 192.168.2.1
(mycondapy310) PS D:\Developer\radarsensing> ssh root@192.168.2.1
# fw_printenv attr_name
attr_name=compatible
# fw_printenv attr_val
attr_val=ad9364
# fw_printenv compatible
compatible=ad9364
# fw_printenv mode
mode=1r1t
# fw_setenv attr_name compatible
# fw_setenv attr_val ad9361
# fw_setenv compatible ad9361
# fw_setenv mode 2r2t
# reboot
```

Check the current mode:
```
(mycondapy310) PS D:\Developer\radarsensing> ssh-keygen -R 192.168.2.1
(mycondapy310) PS D:\Developer\radarsensing> ssh root@192.168.2.1
# fw_printenv attr_name
attr_name=compatible
# fw_printenv attr_val
attr_val=ad9361
# fw_printenv compatible
compatible=ad9361
# fw_printenv mode
mode=2r2t
$ exit
```

Run the test code for SDR:
```bash
(mycondapy310) PS D:\Developer\radarsensing> python .\sdradi\pysdr.py #transmitting a QPSK signal in the 915 MHz band, receiving it, and plotting the PSD
python sdradi/myad9361.py #perform transmit and plot the spectrum
```
### Getting started for Linux

Install the Linux driver from [link](https://wiki.analog.com/university/tools/pluto/drivers/linux)
  * In order to access some USB functions without root privileges, it's recommended to install the PlutoSDR or ADALM2000 udev rules. Download and install this [package](https://wiki.analog.com/_media/university/tools/pluto/drivers/plutosdr-m2k-udev_1.0_amd64.deb)

```bash
$ wget https://wiki.analog.com/_media/university/tools/pluto/drivers/plutosdr-m2k-udev_1.0_amd64.deb
$ sudo dpkg -i plutosdr-m2k-udev_1.0_amd64.deb
$ sudo service udev restart
#plug the device to the USB
$ sudo dmesg | less #Display All Messages from Kernel Ring Buffer, enter q to exit
    You can see usb 1-5: Product: PlutoSDR (ADALM-PLUTO)
$ sudo kermit -l /dev/ttyACM0 -b 115200
    Enter 'c', then type 'root' in pluto login, and 'analog' for password to enter into pluto
    $ uname -a
    Linux pluto 5.15.0-175882-ge14e351533f9 #1 SMP PREEMPT Fri Nov 17 10:23:58 CET 2023 armv7l GNU/Linux
    $ ifconfig
        usb0 inet addr:192.168.2.1
$ mount | grep -i pluto
$ ls -l /media/lkk/PlutoSDR/
$ firefox /media/lkk/PlutoSDR/info.html
$ /sbin/ifconfig
$ ip addr show
    enx00e022e5bbdd inet 192.168.2.10: is the emulated network device for USB
pluto login: root
Password: analog
```
The [dmesg](https://phoenixnap.com/kb/dmesg-linux) command is a Linux utility that displays kernel-related messages retrieved from the kernel ring buffer. The ring buffer stores information about hardware, device driver initialization, and messages from kernel modules that take place during system startup.. You can also monitor the kernel ring buffer in real-time using the --follow option `sudo dmesg --follow`. The option instructs the command to wait for new messages related to hardware or kernel modules after system startup. When searching for specific issues or hardware messages, pipe the dmesg output into grep to search for a particular string or pattern: `sudo dmesg | grep -i usb`.

Install IIO devices: [link](https://wiki.analog.com/university/tools/pluto/drivers/linux)

```bash
$ sudo apt-get install libiio-utils
$ iio_info -n 192.168.2.1 | grep device #find the IIO devices
IIO context has 6 devices:
	iio:device0: ad9361-phy
    iio:device1: xadc
    iio:device2: one-bit-adc-dac
    iio:device3: cf-ad9361-dds-core-lpc (buffer capable)
    iio:device4: cf-ad9361-lpc (buffer capable)
    iio:device5: adi-iio-fakedev
$ iio_readdev -n 192.168.2.1 -s 64 cf-ad9361-lpc | hexdump -x #Read from an IIO device buffer
```

Connect to the device and test the code:
```bash
$ iio_info -n pluto.local
$ iio_info -u ip:pluto.local
$ iio_info -u ip:192.168.2.1
$ ssh root@192.168.2.1
#password: analog
$ python sdradi/pysdr.py
```

## SSH Access
Using POE to power the Mobile Node, i.e., connect the POE cable to the Raspberry Pi Ethernet port with POE hat. The SDR radio is connected to the Raspberry Pi via USB; the Raspberry Pi itself will be served as the analog phaser. The POE will provide all the power to these devices. We can connect to the Mobile Node (i.e., Raspberry Pi) via host device (Mac, Linux or Windows) in the same network:

```bash 
sudo apt install nmap
ipconfig #check the current ip range
nmap -sn 192.168.86.0/24 #scan IP in the current network, nmap -sn 192.168.1.0/24 if the current network IP is 192.168.1.73
ssh analog@192.168.1.69 #password: analog
#second option
ssh analog@phaser
analog@phaser:~ $ ifconfig
    eth0: 192.168.1.67
    eth1: 192.168.2.10
    wlan0: 192.168.1.69
```

Check the analog devices:
```bash
analog@phaser:~ $ iio_attr -a -C fw_version #it will show multiple devices
analog@phaser:~ $ iio_info -u ip:phaser.local #show the Raspberry Pi phaser information
analog@phaser:~ $ iio_info -u ip:192.168.1.67 #use the IP to show the phaser information
analog@phaser:~ $ iio_info -u ip:192.168.2.10 #same information
analog@phaser:~ $ iio_info -u ip:192.168.1.69 #same information
analog@phaser:~ $ iio_info -u ip:phaser.local:50901 #show the SDR information (actual connection is USB)
    hw_model: Analog Devices PlutoSDR Rev.C (Z7010-AD9361)
    uri: ip:phaser.local
	ip,ip-addr: 192.168.2.10
    IIO context has 4 devices:
analog@phaser:~ $ iio_readdev -u ip:pluto.local -B -b 65768 cf-ad9361-lpc
    Throughput: 24 MiB/s
analog@phaser:~ $ iio_readdev -u ip:phaser.local:50901 -B -b 65768 cf-ad9361-lpc
    Throughput: 21 MiB/s
```

# Processing Code

Run the test code for SDR:
```bash
(mycondapy310) PS D:\Developer\radarsensing> python .\sdradi\pysdr.py #transmitting a QPSK signal in the 915 MHz band, receiving it, and plotting the PSD
python sdradi/myad9361.py #perform transmit and plot the spectrum
```

Newly added `myad9361class.py` that put all sdr related code into one class. Run the following code to test the SDR class and perform signal detection
```bash
python sdradi/myad9361class.py
```
The result figure is shown as:
![correctionresults](../imgs/correctionresults.png "Receiver Correction results")

# UI Part
Test mayavi:
```bash
pip install PySide6
python sdrpysim/testmayavi.py
```

Test pyqt:
```bash
python sdrpysim/testpyqt56side6.py #Runtime="QT6"#"QT6" works in windows
```

Run Radar App based on local dataset:
```bash
python .\sdrpysim\pyqt6app.py
```