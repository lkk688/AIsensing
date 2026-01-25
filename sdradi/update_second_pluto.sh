#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

FW_FILE=~/pluto_drivers_build/pluto.frm
MOUNT_POINT="/media/lkk/PlutoSDR"

echo -e "${GREEN}=== Updating Second PlutoSDR ===${NC}"

if [ ! -f "$FW_FILE" ]; then
    echo -e "${RED}Firmware file not found at $FW_FILE${NC}"
    echo "Please run reset_drivers.sh first to download it."
    exit 1
fi

if [ -d "$MOUNT_POINT" ]; then
    echo "Found device at $MOUNT_POINT"
    read -p "Do you want to flash 'pluto.frm' to this device? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Copying firmware..."
        cp "$FW_FILE" "$MOUNT_POINT/"
        sync
        echo -e "${GREEN}Firmware copied.${NC}"
        echo "Ejecting..."
        sudo eject "$MOUNT_POINT"
        echo -e "${GREEN}Please UNPLUG and REPLUG this second PlutoSDR.${NC}"
    fi
else
    echo -e "${RED}Second PlutoSDR not found at $MOUNT_POINT${NC}"
    echo "Checking for other potential mounts..."
    lsblk | grep "Pluto"
fi
