
# NVIDIA Holoscan Sensor Bridge with FPGA

Technical documentation, setup instructions, and implementation details for working with the **NVIDIA Holoscan Sensor Bridge** based on a **Lattice ECP5 FPGA** platform. The system enables high-speed, low-latency sensor data streaming—such as from ADCs and RF transceivers—to NVIDIA GPUs via 10GbE, facilitating real-time processing pipelines for edge AI and medical applications.

## 📦 Overview

NVIDIA Holoscan is a modular, low-latency data processing framework designed for sensor AI workloads. The **Sensor Bridge** is a companion hardware system designed to ingest sensor data (e.g., from ADI RF frontends via FMC connectors), buffer and format it in an FPGA, and stream it to NVIDIA GPUs over 10Gb Ethernet.

This README outlines:

- Setup and usage of the Holoscan Sensor Bridge hardware
- FPGA IP overview and customization
- ADC/transceiver data acquisition from FMC to GPU
- Links to development resources and open-source repositories

---

## 🚀 System Architecture

```text
+-----------------+     FMC      +------------------+     10GbE      +------------------+
| ADI Transceiver | <==========> | Lattice FPGA     | <===========> | NVIDIA GPU Host  |
| (ADC/DAC)       |              | (Sensor Bridge)  |                | (Holoscan SDK)   |
+-----------------+              +------------------+                +------------------+
```

Data Flow:
	1.	Sensor Input: ADC data (e.g., from Analog Devices AD-FMCDAQ2) is input through FMC interface.
	2.	FPGA Processing: FPGA captures data via FMC interface, optionally applies pre-processing (decimation, reordering), and buffers it in internal BRAM/DDR.
	3.	10GbE Transmission: Data is packetized and streamed out using the onboard 10GbE MAC/PHY via UDP.
	4.	GPU Reception: NVIDIA Holoscan SDK receives the data in a GXF pipeline and forwards it to GPU memory for real-time processing.

⸻

🔧 Hardware Setup

Refer to NVIDIA Holoscan Sensor Bridge hardware guide:

📄 Sensor Bridge Hardware Setup

Key Components:
	•	FPGA Board: Lattice ECP5-based development board
	•	FMC Interface: Connects to Analog Devices ADC boards
	•	10GbE PHY: Interfaces to host machine or Jetson AGX Orin
	•	Power Supply & Clock: External supplies and clocking via FMC or onboard PLL

⸻

🧠 FPGA IP Overview

Refer to the official IP list:
📄 Sensor Bridge IP Catalog

Notable IP blocks:

IP Core	Description
FMC Interface	Captures LVDS parallel or serial (JESD204B) data from ADCs
Data Formatter	Applies bit reordering, channel multiplexing, metadata tagging
10GbE MAC/PHY	Handles UDP encapsulation and 10GbE transmission
AXI Stream Bridge	Connects core logic to external DRAM or streaming interface
Clock Generator	Provides PLL-based clocks for JESD or LVDS capture
Control Register	Memory-mapped registers for configuration and runtime control


⸻

🔄 ADC/FMC Integration Guide

Step-by-Step: Getting ADC Data to GPU
	1.	Hardware Connection
	•	Connect an ADI FMC-compatible ADC (e.g., AD-FMCDAQ2) to the Sensor Bridge FMC port.
	•	Ensure proper clocking (either from the ADC or provided to it).
	2.	FPGA Configuration
	•	Build the bitstream using Lattice Radiant or Diamond with IP cores configured for your ADC.
	•	Configure JESD204B or parallel interface for the ADC on FMC.
	3.	UDP Data Stream
	•	Configure packet size, sample format, and stream rate in the Control Register block.
	•	Data is transmitted over 10GbE using a UDP stream formatted as expected by the Holoscan operator.
	4.	Host Machine
	•	On the Jetson AGX Orin or x86 GPU host:
	•	Use Holoscan SDK’s VideoStreamReplayer or UDPReceiverOp to capture packets.
	•	Connect to downstream operators or models via the GXF graph.

⸻

🧪 Sample Projects

✅ FPGA Sample Design
	•	Includes:
	•	FMC ADC interface setup (parallel and JESD204)
	•	Simple data formatter and packetizer
	•	UDP MAC and PHY config
	•	Example from official guide:
	•	FPGA Samples - NVIDIA Docs

✅ Host Code (C++/Python)
	•	Holoscan GXF app pipeline: receives UDP packets, converts to GPU buffers
	•	Uses:
	•	holoscan::ops::UdpReceiverOp
	•	holoscan::ops::TensorReformatterOp
	•	Optional: Extend with AI inference models or visualization

⸻

📚 Resources

Official Documentation
	•	NVIDIA Holoscan Sensor Bridge Docs
	•	NVIDIA Holoscan SDK GitHub

Development Tools
	•	Lattice Radiant IDE (for ECP5 synthesis & IP integration)
	•	Vivado (if adapting to Xilinx-based bridge)
	•	Wireshark (UDP packet analysis)
	•	Holoscan CLI & SDK

⸻

## Problems

Unlike larger FPGA vendors like Xilinx and Intel (Altera), Lattice:
	•	Does not provide JESD204B/C IP cores as part of its default IP catalog for the ECP5 or CrossLink-NX families.
	•	Has no JESD204 IP mentioned in official Radiant or Diamond toolchains.

According to NVIDIA’s documentation, the Holoscan FPGA samples and IPs target parallel or LVDS interfaces, not JESD204.

✅ Workarounds / Community & Third-Party Solutions

To implement JESD204 interfaces with Lattice FPGAs, you have three options:
	1.	Use ADCs with Parallel LVDS Interface Instead
	•	ADI (Analog Devices) provides FMC boards (e.g., AD9645, AD7961) that use parallel LVDS or CMOS output instead of JESD204.
	•	The Holoscan Sensor Bridge does support LVDS interfacing, and NVIDIA provides IPs and examples for those in the Lattice environment.

Use External JESD-to-LVDS Bridge
	•	Some system designs use an intermediate ASIC or FPGA to convert JESD204 output from ADC to a parallel or simpler interface the Lattice FPGA can handle.
	•	This adds complexity and latency but is a valid approach.

---

## 📂 Using Xilinx ZCU102
Using the Xilinx ZCU102 board to interface with an ADI ADC via JESD204, and stream the captured data to an NVIDIA GPU via 10GbE, is very feasible and robust — especially since Xilinx provides JESD204 IP and ADI provides verified JESD204 reference designs.

Here’s a structured guide to building such a pipeline:

⸻

🔧 System Overview

+----------------------+     JESD204B     +--------------------+     10GbE      +------------------+
| ADI FMC ADC Board    | <=============> | Xilinx ZCU102       | <===========> | NVIDIA GPU Host  |
| (e.g., AD9680, etc.) |                 | (JESD + UDP Stream) |                | (UDP Receiver)   |
+----------------------+                 +--------------------+                +------------------+


⸻

🧱 Step 1: Hardware Requirements
	•	ZCU102 Evaluation Board (Zynq UltraScale+)
	•	ADI JESD204 ADC FMC board (e.g., AD9680, AD-FMCDAQ2, AD9208)
	•	10GbE SFP+ module and cable
	•	NVIDIA Jetson AGX Orin or x86 PC with GPU + 10GbE NIC
	•	Optionally: ADI Clocking Board (e.g., AD9528) to provide JESD clocks

⸻

⚙️ Step 2: Setup JESD204B on ZCU102

Option A: Use ADI Reference Design (Recommended)

ADI provides full JESD204-compatible designs for Xilinx platforms via:
	•	ADI HDL GitHub: https://github.com/analogdevicesinc/hdl
	•	Supported Boards: AD-FMCDAQ2, AD9680, AD9208 + ZCU102

To build the design:
	1.	Clone the ADI HDL repo:

git clone https://github.com/analogdevicesinc/hdl.git
cd hdl/projects/daq2/zcu102


	2.	Use Make or Vivado GUI to build the project:

make


	3.	This design:
	•	Instantiates Xilinx JESD204 IP (PHY + LINK + Transport)
	•	Receives data from ADC via FMC
	•	Maps data into AXI-stream or memory

⸻

🌐 Step 3: Add 10GbE UDP Streaming Logic

Option A: Use Xilinx CMAC + UDP Packetizer (Vivado IP + HLS or RTL)
	1.	Insert CMAC IP
	•	Use the 10G/25G CMAC Subsystem or 10G Ethernet Subsystem IP
	•	Connect to your data source (AXI stream from JESD receiver)
	2.	Add UDP Packetizer
	•	Custom logic or HLS:
	•	Wrap JESD samples in UDP packets
	•	Add appropriate headers (Ethernet + IP + UDP)
	•	Inject into CMAC’s TX stream
	3.	MAC Address and IP Configuration
	•	Use AXI-lite to set IP, port, MAC in CMAC wrapper
	•	Use either static or dynamically configured UDP parameters

⸻

🖥️ Step 4: Host-Side GPU UDP Receiver

Option A: Use Holoscan SDK or Custom GStreamer Pipeline

On the host (Jetson or x86):
	1.	Set up UDP capture:
	•	With Holoscan:

class UdpReceiverOp(Operator):
    ...


	•	Or with GStreamer:

gst-launch-1.0 udpsrc port=9000 ! application/x-raw,format=... ! appsink


	2.	Forward packets to GPU:
	•	Use CUDA buffer or Holoscan MemoryResource to zero-copy transfer

⸻

📂 FPGA Design Block Diagram

[FMC ADC]
   |
[JESD204 PHY + LINK + Transport]
   |
[AXI-Stream Interface]
   |
[Custom Packetizer (UDP/IP)]
   |
[10G Ethernet CMAC]
   |
[SFP+ PHY (GTY transceivers)]

Optional additions:
	•	DDR buffering for burst mode
	•	Timer IP for timestamping
	•	PTP synchronization

⸻

From ADI:
	•	ADI HDL GitHub
	•	ADI Wiki: AD-FMCDAQ2 + ZCU102
	•	ADI JESD204 Framework

From Xilinx:
	•	JESD204 IP Overview
	•	CMAC IP Product Guide
	•	10G Ethernet Subsystem

⸻
	•	Use GTY transceivers on ZCU102 for JESD and Ethernet.
	•	Enable Xilinx Debug Bridge + ILA for debugging JESD alignment and link status.
	•	Ensure correct lane alignment, sysref timing, and clock tree routing when using JESD204B.
	•	Validate data integrity on the GPU using checksum or header verification.

## 📂 Intel FPGA

Summary Workflow Using Intel Agilex 5

[ADI ADC FMC Board]
   ↓ JESD204B/C
[Intel Agilex 5 FPGA]
   ↓ AXI/ST or Avalon-ST
[UDP Packetizer (custom IP)]
   ↓
[Intel 10GbE MAC + PHY]
   ↓
[10GbE SFP+]
   ↓
[NVIDIA GPU Host]


⸻

🧱 Step 1: Hardware Requirements
	•	Intel Agilex 5 Development Kit (e.g., E-Series with H-tile or R-tile)
	•	ADI JESD204B ADC board (e.g., AD9680, AD9208, AD-FMCDAQ2)
	•	Clocking board (ADI AD9528) if required
	•	SFP+ cage + 10GbE module
	•	GPU-enabled host (Jetson AGX Orin or x86 PC with NIC)

⸻

⚙️ Step 2: JESD204B Setup on Agilex 5

Intel provides JESD204B IP in Quartus Pro with full support on Agilex.

➤ IP Configuration
	1.	Open Intel Quartus Pro
	2.	Create a new Platform Designer system
	3.	Add:
	•	JESD204B IP (Intel) as receiver (RX)
	•	Reference clock + SYSREF inputs
	•	ADI FMC pins mapped to transceivers (match lane rate & alignment)
	4.	Enable internal PLL or use external reference via AD9528

➤ Clocking Notes
	•	Lane rate: e.g., 6.25 Gbps for AD9208
	•	SYSREF must align to LMFC boundaries
	•	Use transceiver tiles (F-Tile or H-Tile)

⸻

📦 Step 3: Capture ADC Data
	1.	Connect JESD204 RX IP to:
	•	Avalon-ST or AXI-ST interface
	•	Align lanes (lane0, lane1…)
	•	Pack samples (e.g., 16-bit, 12-bit) into 64/128-bit words
	2.	Optionally buffer in:
	•	Dual-port RAM
	•	FIFO
	•	DDR controller (if burst-mode)

⸻

🌐 Step 4: 10GbE Transmission

Intel provides the 10G Ethernet MAC + PCS/PMA PHY IP (or use the 10GbE Subsystem).

➤ Options:
	•	10GBASE-R MAC (PCS+PMA) for SFP+ optical or DAC cable
	•	Use GMII/MII interface or Avalon-ST
	•	Add UDP stack via:
	•	Custom RTL
	•	Open-source IP (e.g., LiteEth, HLS-generated UDP core)

➤ UDP Packetizer (Custom)
	•	Create a module that:
	•	Adds Ethernet/IP/UDP headers
	•	Appends ADC samples
	•	Computes checksums (optional)
	•	Sends out Avalon-ST to 10GbE MAC

⸻

🖥️ Step 5: Host-Side Receiver (GPU)

On the GPU host (Jetson or x86):
	1.	Use NVIDIA Holoscan SDK or a custom GStreamer/CUDA pipeline to receive and process UDP packets
	2.	Optionally:
	•	Reconstruct samples
	•	Visualize
	•	Forward to ML pipeline

⸻

 Intel Platform Designer Block Diagram (Simplified)

[FMC ADC]
   ↓
[JESD204B RX IP]
   ↓
[Data Aligner + Packer]
   ↓
[UDP Packet Builder]
   ↓
[10G Ethernet MAC (Avalon-ST)]
   ↓
[SFP+ PHY (H-tile or F-tile)]


⸻

📚 Key Resources

Intel:
	•	JESD204B Intel FPGA IP User Guide: link
	•	10G Ethernet MAC User Guide: link
	•	Quartus Pro Design Examples (available via Design Store)

ADI:
	•	FMC boards: https://wiki.analog.com/resources/eval/user-guides/fmc
	•	JESD204 User Guide: https://wiki.analog.com/resources/fpga/peripherals/jesd204

Open Source (Optional UDP Stack):
	•	LiteEth
	•	HLS UDP/IP Generator

## Modify Existing Arria10 FPGA for Agilex FPGA
To modify the existing ADI HDL project (e.g., ADRV9009 on Intel Arria 10) to target a new Intel Agilex FPGA with 10GbE support:

⸻

🧭 Goal

Adapt an ADI HDL project like:

hdl/projects/adrv9009/a10gx/

To:

hdl/projects/adrv9009/agilex/

With:
	•	Target FPGA: Intel Agilex (E-Series or F-Series)
	•	JESD204B IP remapped for Agilex transceivers
	•	Added or updated 10GbE MAC/PHY (Avalon-ST or custom)
	•	Platform-specific pinouts, clocks, and constraints

⸻

🔧 Step-by-Step Guide

✅ 1. Clone ADI HDL repo

git clone https://github.com/analogdevicesinc/hdl.git
cd hdl/projects/adrv9009


⸻

🔄 2. Create New Project Directory

cp -r a10gx agilex
cd agilex

Update:
	•	Makefile
	•	Platform references: replace a10gx with agilex
	•	Board-specific config (e.g., system_project.tcl, system_bd.tcl, system_constr.sdc)

⸻

⚙️ 3. Update Platform IP Definitions

Edit Platform Designer (Qsys) or TCL IP scripts to:
	•	Replace Arria 10 transceiver PLLs and PHYs with Agilex equivalents
	•	Update JESD204 PHY interface (Agilex may use F-Tile or E-Tile PHY IP)

➤ IP Changes:

Arria 10	Agilex Equivalent
Arria 10 Transceiver PHY	Agilex F-Tile PMA+PCS
Clocking Wizard	Agilex Clock Control Block
JESD204B Intel IP	Rebuild for Agilex PHY and JESD mode
DDR4 Controller	Adjust for memory topology


⸻

🧠 4. Update JESD204 Interface

From:

set_instance_parameter "jesd_rx" "DeviceFamily" "Arria10"

To:

set_instance_parameter "jesd_rx" "DeviceFamily" "Agilex"

Then:
	•	Regenerate JESD204B IP for Agilex PHY and speed (e.g., 10 Gbps)
	•	Connect to appropriate transceiver channels (GTX/FTile)

⸻

🌐 5. Add or Update 10GbE MAC (Intel)

You can use Intel’s 10G Ethernet MAC + PCS/PMA IP:

Add in Platform Designer:
	•	10G Ethernet MAC with Avalon-ST
	•	10G PCS/PMA (PHY) or direct connection to Agilex transceiver tile

Connect:
	•	Data input: AXI/Avalon-ST from JESD204 capture FIFO
	•	Clock input: 156.25 MHz refclk from SFP+ clock
	•	Control: Avalon-MM interface or custom control module

⸻

📦 6. Add UDP Packetizer Module

Use or create:
	•	Simple Verilog HDL that:
	•	Collects ADC samples
	•	Adds Ethernet + IP + UDP headers
	•	Packs into 1500-byte payloads
	•	Computes UDP length + checksum (optional)

Connect packetizer output to 10G MAC Avalon-ST TX

⸻

📄 7. Modify Constraint Files (SDC, QSF)

Update pin assignments for:
	•	FMC connector (ADC interface)
	•	SYSREF, REFCLK, SYNC
	•	SFP+ TX/RX pairs
	•	Clocking (PLL inputs, refclk)

⸻

🧪 8. Test in Quartus Pro

cd hdl/projects/adrv9009/agilex
make

Test plan:
	•	Run Signal Tap to check JESD204 link status (ILAS, CGS)
	•	Wireshark on host to verify UDP packet output
	•	Loopback or test generator mode on ADRV9009 to validate end-to-end

⸻

📁 File Tree Overview

hdl/projects/adrv9009/agilex/
├── Makefile
├── system_bd.tcl            # Top-level Platform Designer script
├── system_project.tcl       # Quartus project script
├── system_constr.sdc        # Constraints (clocks, timing)
├── custom_udp_packetizer.v  # New module (to be created)
└── platform_ip/             # JESD204B, 10GbE MAC, clock IP


