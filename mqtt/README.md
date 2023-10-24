# Cloud-Based Neural Network Inference for Automated Vehicles

This project aims to implement and evaluate two different methodologies for moving neural network inference from automated vehicles to connected cloud servers. We are using ROS (Robot Operating System) and MQTT (Message Queuing Telemetry Transport) for communication.

# MQTT
## Code Overview

The code in this project consists of two ROS nodes implemented as Python classes:

1. `VehicleNode`: This node simulates a vehicle by publishing images from a camera (read from a .tiff file) and subscribing to segmented images from the cloud.

2. `CloudNode`: This node represents a cloud server. It subscribes to the camera images published by the vehicle, performs inference (simulated with a sleep function), and publishes the segmented images back to the vehicle.

## Usage

### Prerequisites

- [ROS](http://wiki.ros.org/ROS/Installation) installed on your machine.
- [Paho MQTT](https://pypi.org/project/paho-mqtt/) Python client installed (`pip install paho-mqtt`).
- Python Imaging Library (PIL) installed (`pip install pillow`).

### Running the nodes

1. Save the Python scripts as separate `.py` files within the `scripts` directory of your ROS package.

2. Make each Python script executable:

```bash
chmod +x vehicle_node.py
chmod +x cloud_node.py
```

3. Run each node with the rosrun command:
```bash
rosrun your_package vehicle_node.py
rosrun your_package cloud_node.py
```

## Setting up the MQTT Broker

We recommend using Mosquitto as your MQTT broker. Mosquitto is an open-source message broker that implements the MQTT protocol. It can be easily set up as a Docker container, but you can also install it directly on your machine.

### Using Docker

1. Install [Docker](https://docs.docker.com/get-docker/) on your machine.

2. Run the following command to launch Mosquitto:

```bash
docker run --rm -d -p 1883:1883 --name mosquitto eclipse-mosquitto
```
This will start a new Mosquitto instance as a detached Docker container. The -p 1883:1883 argument maps the broker's port (1883) to your machine's port 1883.

### Without Docker
Install Mosquitto on your machine. The process varies depending on your OS, but for Ubuntu, you can use the following commands:

```bash
sudo apt update
sudo apt install mosquitto mosquitto-clients
```
Start the Mosquitto service:

```bash
sudo systemctl start mosquitto
```
To check the status of the service, use:
```bash
sudo systemctl status mosquitto
```

Update to file cloud_node.py Jul 15-The file for now reads from the rosbag 'left_camera_templergraben.bag', performs inference using 'frozen_graph_runner.py' and sends the segmented images to the MQTT topic /segmented_images. 