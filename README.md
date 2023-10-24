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


# TensorFlow Serving

This repository is part of the ACDC Research Project and focuses on TF-Serving related progress.

## Tasks for now

- [X] Add ROSBag Integration
- [ ] Benchmark
## Features

- Run TensorFlow Serving for model inference.
- Perform vehicle image segmentation using deep learning models.
- Support for both CPU and GPU environments.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Docker is installed on your machine.
- For GPU usage, ensure NVIDIA driver and NVIDIA Container Toolkit are installed.
- Access to the IKA Workstation or a cloud station via SSH.


## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/vehicle-cloud-inference.git

2. Navigate to the project directory:
   ```bash
   cd vehicle-cloud-inference
   ```

3. Install the required Python packages(preferably in a virtualenv/conda env):
   ```bash
   pip install -r requirements.txt
   ```

## Models Used

1. [best_weights_e_00231_val_loss_0.1518.zip](https://git.rwth-aachen.de/ika/acdc-research-project-ss23/acdc-research-project-ss23/uploads/e5bdaf3b7aa6d2b59bbd098e55eb079c/best_weights_e_00231_val_loss_0.1518.zip)
2. [mobilenetv3_large_os8_deeplabv3plus_72miou.zip](https://git.rwth-aachen.de/ika/acdc-research-project-ss23/acdc-research-project-ss23/uploads/3f73d5bd57acc307182278c0e0449650/mobilenetv3_large_os8_deeplabv3plus_72miou.zip)
   
## Usage

### Running TensorFlow Serving Docker Container on your PC (Will automatically pull if Docker image not found on Device)

- For CPU:
  ```bash
  docker run -p 8500:8500 -p 8501:8501 --mount type=bind,source={full_path}/model/mobilenetv3_large_os8_deeplabv3plus_72miou/,target=/models/mobilenet/1/ -e MODEL_NAME=mobilenet -t tensorflow/serving
  ```
- For GPU(ensure NVIDIA Toolkit is installed)
  ```bash
  docker run --gpus all -p 8500:8500 -p 8501:8501 --mount type=bind,source={full_path}/model/mobilenetv3_large_os8_deeplabv3plus_72miou/,target=/models/mobilenet/1/ -e MODEL_NAME=mobilenet -t tensorflow/serving:2.11.1-gpu

  ```
### Running Inference

After starting the Docker container, execute the following command to perform inference:
```bash
python segmentation_mobilenet_serving.py -h 

Specify trigger and model_export_path.

options:
  -h, --help            show this help message and exit
  --bag BAG             Path to the Bag file
  --model_export_path MODEL_EXPORT_PATH
                        Path to the model export directory. Make sure the model path matches the one TFServing is serving :)
  --trigger TRIGGER     Trigger for mode (e.g., "grpc" or "rest").

```

### Connecting to IKA Workstation (If you want to run the inference there)
1. Connect to the IKA Workstation using SSH and the RWTH VPN.

2. Pull the required TensorFlow Serving Docker Image: 
   ```bash
   docker pull tensorflow/serving:2.11.1-gpu (if using GPU).
   docker pull tensorflow/serving (if using CPU)
   ```
3. You can run the similar commands as in [section 2](#running-tensorflow-serving-docker-container-on-your-pc)
4. Enable port forwarding to port 8501(HTTPS REST) or port 8500 (gRPC).
5. Update the URL in segmentation_mobilenet_serving.py to match the workstation's URL
   (For example http://i2200049.ika.rwth-aachen.de:8501/v1/models/mobilenet:predict in the URL variable)
6. Run segmentation_mobilenet_serving.py to view results.
   
## Note

-  Ensure Docker and NVIDIA Toolkit (if using GPU) are properly set up before running.
-  Update file paths and URLs as required.
-  Port forwarding may be necessary for remote inference.
-  We are opening both 8500 and 8501 ports for gRPC and HTTPS respectively. Use gRPC for faster inference

## Credits

This project is part of the ACDC Research Project at RWTH Aachen University.