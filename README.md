# Keg Counting & QR Detection System

## Overview

The **Keg Counting & QR Detection System** is a computer vision–based solution for automated keg counting, QR code detection, and pallet management in brewery and industrial environments. The system uses **YOLO-based object detection**, **multi-method QR recognition**, and **cloud synchronization** to support both manual and automated workflows through a simple operator HMI.

## Key Features

- Real-time keg detection and counting
- QR code detection using multiple methods
- Pallet lifecycle tracking
- Cloud API integration
- Manual and auto capture modes
- Recovery and retry handling
- Basic reporting and logging

## Supported Platforms

- Ubuntu 20.04+
- Windows 10+
- NVIDIA Jetson / CUDA-capable systems
- Python 3.8+

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-org>/keg-system.git
cd keg-system
````

Create and activate virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

(Windows: `venv\Scripts\activate`)

Install system dependencies (Ubuntu):

```bash
sudo apt-get update
sudo apt-get install -y libzbar0 ffmpeg libgl1-mesa-glx
```

Install Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Application

Start the system:

```bash
python main.py
```

This launches the camera, detection pipeline, operator HMI, and cloud synchronization.

### One-Click Execution

**Linux (Jetson):**
Run the shell script:
```bash
./Start_palletization.sh
```
(Ensure it is executable with `chmod +x Start_palletization.sh` first)

## Version

* Version: 2.0.0
* Last Updated: January 2026

## License

Proprietary – All rights reserved.

