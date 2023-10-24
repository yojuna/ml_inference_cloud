#!/usr/bin/python3


import paho.mqtt.client as mqtt
import numpy as np
import cv2
import matplotlib.pyplot as plt

MQTT_BROKER_IP = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_SUB_TOPIC = "/vehicle_camera"

def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker with code:", rc)
    client.subscribe(MQTT_SUB_TOPIC)

def on_message(client, userdata, msg):
    try:
        np_arr = np.frombuffer(msg.payload, np.uint8)
        print("Received image data length:", len(np_arr))
        if len(np_arr) == 0:
            print("Empty image data received")
            return

        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

    except Exception as e:
        print("Error:", e)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(MQTT_BROKER_IP, MQTT_BROKER_PORT, 60)
client.loop_forever()