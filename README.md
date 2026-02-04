# Chemist Eye IR Stations

This repository contains the software and hardware configuration for the **Chemist Eye Infrared (IR) monitoring stations**, designed to detect elevated temperatures and potential fire hazards in self-driving laboratories.

These stations complement the RGB-D perception layer by providing thermal awareness to the Chemist Eye safety architecture.

---

## System Overview

The IR station streams thermal images and temperature statistics to a central ROS Master for analysis and decision-making.

When temperatures exceed predefined thresholds, the system can:

* Notify laboratory users
* Trigger audible alarms
* Reroute mobile robots away from hazardous areas

All commands are executed remotely via SSH from the ROS master.

---

## Repository Structure

```
chemist_eye_ir/
│
├── streamer.py
```

### Key Script

**`streamer.py`**
Connects to the thermal camera, processes temperature frames, applies a colormap for visualization, and streams encoded images together with the maximum detected temperature to the Chemist Eye server.

---

## Bill of Materials

Hardware configuration used in the study:

* **Raspberry Pi 5** running Raspberry Pi OS
* **MI48 long-wave infrared thermal camera**
* Tripod or custom mount for flexible placement
* Standard networking and power supply

The camera operates approximately between **20 °C and 400 °C**, making it suitable for monitoring common laboratory equipment such as hot plates and reaction stations.

---

## How It Works

1. The MI48 thermal camera captures temperature data.
2. Frames are filtered and normalized for clarity.
3. A colormap is applied to improve interpretability.
4. The processed image and maximum temperature are transmitted to the Chemist Eye server.
5. If a threshold is exceeded, the ROS master initiates safety actions.

Threshold temperatures are fully configurable depending on the experimental setup.

---

## Installation

Ensure the thermal camera drivers and required libraries are available on the device.

---

## Running the Streamer

```bash
python streamer.py
```

Optionally specify the streaming FPS:

```bash
python streamer.py 15
```

---

## Deployment Notes

* Stations can be placed near reaction setups, inside fume hoods, or close to heat-generating equipment.
* Fixed IP networking is recommended.
* SSH access must be enabled for remote orchestration.

---

## Related Repositories

⚠️ This repository is part of the Chemist Eye ecosystem.
For the full system architecture, please visit:

[https://github.com/FranciscoMunguiaGaleano/chemist_eye](https://github.com/FranciscoMunguiaGaleano/chemist_eye)

RGB-D stations:

[https://github.com/FranciscoMunguiaGaleano/chemist_eye_rgbd](https://github.com/FranciscoMunguiaGaleano/chemist_eye_RGB-D)

---

