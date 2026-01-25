#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== PlutoSDR Driver & Firmware Reset Script ===${NC}"
echo "This script will:"
echo "1. Check/Install dependencies for libiio"
echo "2. Build libiio from source (latest)"
echo "3. Reinstall Python bindings"
echo "4. Update PlutoSDR Firmware to v0.39"
echo ""

# 1. Dependency Check
echo -e "${YELLOW}[1/4] Checking Dependencies...${NC}"
MISSING_DEPS=""
dpkg -s cmake >/dev/null 2>&1 || MISSING_DEPS="$MISSING_DEPS cmake"
dpkg -s bison >/dev/null 2>&1 || MISSING_DEPS="$MISSING_DEPS bison"
dpkg -s flex >/dev/null 2>&1 || MISSING_DEPS="$MISSING_DEPS flex"
dpkg -s libxml2-dev >/dev/null 2>&1 || MISSING_DEPS="$MISSING_DEPS libxml2-dev"
dpkg -s libcdk5-dev >/dev/null 2>&1 || MISSING_DEPS="$MISSING_DEPS libcdk5-dev"
dpkg -s libaio-dev >/dev/null 2>&1 || MISSING_DEPS="$MISSING_DEPS libaio-dev"
dpkg -s libusb-1.0-0-dev >/dev/null 2>&1 || MISSING_DEPS="$MISSING_DEPS libusb-1.0-0-dev"
dpkg -s libavahi-client-dev >/dev/null 2>&1 || MISSING_DEPS="$MISSING_DEPS libavahi-client-dev"
dpkg -s libavahi-common-dev >/dev/null 2>&1 || MISSING_DEPS="$MISSING_DEPS libavahi-common-dev"

if [ ! -z "$MISSING_DEPS" ]; then
    echo -e "${RED}Missing dependencies: $MISSING_DEPS${NC}"
    echo "Please run: sudo apt install $MISSING_DEPS"
    echo "Then run this script again."
    exit 1
else
    echo -e "${GREEN}All dependencies found.${NC}"
fi

# 2. Build libiio
echo -e "${YELLOW}[2/4] Building libiio from source...${NC}"
WORK_DIR=~/pluto_drivers_build
mkdir -p $WORK_DIR
cd $WORK_DIR

if [ ! -d "libiio" ]; then
    git clone https://github.com/analogdevicesinc/libiio.git
fi
cd libiio
# Pull latest
git checkout main 2>/dev/null || git checkout master 2>/dev/null
git pull

mkdir -p build && cd build
# Note: Python bindings via CMake often install to system paths, we will install explicitly via pip later
cmake .. -DCMAKE_INSTALL_PREFIX=/usr -DENABLE_PYTHON_BINDINGS=OFF 
make -j$(nproc)

echo "Installing libiio (requires sudo)..."
sudo make install
sudo ldconfig

# 3. Reinstall Python Bindings FROM SOURCE (Crucial for v1.0 support)
echo -e "${YELLOW}[3/4] Reinstalling Python bindings from source...${NC}"

# Install low-level 'iio' module from libiio source
cd $WORK_DIR/libiio/bindings/python
pip install .

# Install high-level 'pyadi-iio' from source
cd $WORK_DIR
if [ ! -d "pyadi-iio" ]; then
    git clone https://github.com/analogdevicesinc/pyadi-iio.git
fi
cd pyadi-iio
git pull
pip install .

# 4. Firmware Update
echo -e "${YELLOW}[4/4] Checking Firmware...${NC}"
FW_VER="v0.39"
FW_FILE="plutosdr-fw-${FW_VER}.zip"
URL="https://github.com/analogdevicesinc/plutosdr-fw/releases/download/${FW_VER}/${FW_FILE}"

# Download if not exists
if [ ! -f "$WORK_DIR/$FW_FILE" ]; then
    echo "Downloading Firmware $FW_VER..."
    wget -O "$WORK_DIR/$FW_FILE" "$URL"
fi

cd $WORK_DIR
unzip -o $FW_FILE

# Detect Mount
PLUTO_MOUNT="/media/lkk/PlutoSDR"
if [ -d "$PLUTO_MOUNT" ]; then
    echo -e "${GREEN}PlutoSDR found at $PLUTO_MOUNT${NC}"
    echo "Current contents:"
    ls $PLUTO_MOUNT
    
    read -p "Do you want to flash 'pluto.frm' to this device? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Copying firmware..."
        cp pluto.frm "$PLUTO_MOUNT/"
        sync
        echo -e "${GREEN}Firmware copied.${NC}"
        echo "Ejecting..."
        sudo eject "$PLUTO_MOUNT"
        echo -e "${GREEN}Please UNPLUG and REPLUG the PlutoSDR. It will blink rapidly while updating.${NC}"
    fi
else
    echo -e "${RED}PlutoSDR volume not found at $PLUTO_MOUNT${NC}"
    echo "If using Dual Plutos, check /media/lkk/PlutoSDR1 manually."
    echo "Firmware file is located at: $WORK_DIR/pluto.frm"
fi

echo -e "${GREEN}=== Done ===${NC}"
