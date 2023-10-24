#!/usr/bin/python3


import rospy
from sensor_msgs.msg import Image
import paho.mqtt.client as mqtt
from PIL import Image as PilImage
import numpy as np
from cv_bridge import CvBridge
import cv2
import matplotlib.pyplot as plt
import time


# MQTT Broker
MQTT_BROKER_IP = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_SSL_BROKER_PORT = 8883

MQTT_CERTS_DIR = "/home/rosuser/ws/ws_mqtt_nodes/vehicle_node/src/mqtt_node/src/"

MQTT_SSL_CLIENT_CERT = MQTT_CERTS_DIR + "cert/client/client-cert.pem"
MQTT_SSL_CLIENT_KEY = MQTT_CERTS_DIR + "cert/client/client-key.pem"
MQTT_SSL_CLIENT_CSR =  MQTT_CERTS_DIR + "cert/client/client-csr.pem"
MQTT_SSL_CLIENT_CA =  MQTT_CERTS_DIR + "cert/ca-cert.pem"
# # testing/vehicle-cloud-inference/mqtt_node/src/cert/client/client-cert.pem
# MQTT_SSL_CLIENT_CERT = "cert/client/client-cert.pem"
# MQTT_SSL_CLIENT_KEY = "cert/client/client-key.pem"
# MQTT_SSL_CLIENT_CSR = "cert/client/client-csr.pem"


MQTT_PUB_CAMERA_TOPIC = "/vehicle_camera"
MQTT_SUB_SEGMENTED_TOPIC = "/segmented_images"


# ROS
ROS_PUB_SEGMENTED_TOPIC = "/segmented_ros_images"
ROS_SUB_CAMERA_TOPIC = "/sensors/camera/left/image_raw"

# class ImageSubscriber:
#     def __init__(self):
#         self.bridge = CvBridge()

#         # MQTT setup
#         self.mqtt_client = mqtt.Client()
#         self.mqtt_client.on_message = self.on_mqtt_message
#         self.mqtt_client.connect(MQTT_BROKER_IP, MQTT_BROKER_PORT, 60)
#         self.mqtt_client.subscribe(MQTT_SUB_SEGMENTED_TOPIC)
#         self.mqtt_client.loop_start()

#         # ROS setup
#         self.ros_pub = rospy.Publisher(ROS_PUB_SEGMENTED_TOPIC, Image, queue_size=10)
#         rospy.init_node('mqtt_image_subscriber', anonymous=True)

#     def on_mqtt_message(self, client, userdata, msg):
#             rospy.loginfo_once("Received segmented image from MQTT")
#             np_arr = np.frombuffer(msg.payload, np.uint8)
#             cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # Seems to work with Jpg format
            
#             # Display image using matplotlib
#             plt.imshow(cv_image)
#             plt.show()

#             # Publish image to ROS topic
#             ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
#             self.ros_pub.publish(ros_image)



# class ImagePublisher:
#     def __init__(self):
#         self.bridge = CvBridge()
        
#         # MQTT setup
#         self.mqtt_client = mqtt.Client()
#         #self.mqtt_client.on_message = self.on_mqtt_message
#         self.mqtt_client.connect(MQTT_BROKER_IP,MQTT_BROKER_PORT,60)
#         self.mqtt_client.publish(MQTT_PUB_CAMERA_TOPIC)
#         self.mqtt_client.loop_start()

#         # ROS setup
#         self.ros_sub = rospy.Subscriber(ROS_SUB_CAMERA_TOPIC,Image,self.callback)
        
#         rospy.init_node('mqtt_image_publisher',anonymous=True)

#     def callback(self,data):
#         try:
#             rospy.loginfo("reading from camera feed")
#             cv_image = self.bridge.imgmsg_to_cv2(data,desired_encoding='bgr8')
#             _,jpeg = cv2.imencode('.jpg',cv_image)
#             self.mqtt_client.publish(MQTT_PUB_CAMERA_TOPIC,jpeg.tobytes())
#             rospy.loginfo("img published to broker at topic %s",MQTT_PUB_CAMERA_TOPIC)
#         except self.bridge.CvBridgeError as e:
#             rospy.logerr(e)


class VehicleNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.ssl_enabled = True

        #setup MQTT Client
        self.mqtt_pub_client = mqtt.Client() # To send vehicle data to the cloud node
        self.mqtt_sub_client = mqtt.Client() # To receive segmented images

        if self.ssl_enabled:
            self.mqtt_pub_client.tls_set(
                certfile=MQTT_SSL_CLIENT_CERT,
                keyfile=MQTT_SSL_CLIENT_KEY,
                ca_certs=MQTT_SSL_CLIENT_CA)

            self.mqtt_sub_client.tls_set(
                certfile=MQTT_SSL_CLIENT_CERT,
                keyfile=MQTT_SSL_CLIENT_KEY,
                ca_certs=MQTT_SSL_CLIENT_CA)


            self.mqtt_pub_client.connect(MQTT_BROKER_IP, MQTT_SSL_BROKER_PORT, 60)
            self.mqtt_sub_client.connect(MQTT_BROKER_IP, MQTT_SSL_BROKER_PORT, 61)
        else:
            self.mqtt_pub_client.connect(MQTT_BROKER_IP, MQTT_BROKER_PORT, 60)
            self.mqtt_sub_client.connect(MQTT_BROKER_IP, MQTT_BROKER_PORT, 61)

        self.mqtt_sub_client.on_message = self.on_mqtt_message #Does the callback job for the subscription of the segmented images

        # List to vehicle's camera (ROS SUB)
        self.ros_sub = rospy.Subscriber(ROS_SUB_CAMERA_TOPIC, Image, self.callback)
        #Callback has the MQTT sending to cloud part

        # Receive data from cloud through MQTT topic
        self.mqtt_sub_client.subscribe(MQTT_SUB_SEGMENTED_TOPIC)
        self.ros_pub = rospy.Publisher(ROS_PUB_SEGMENTED_TOPIC, Image, queue_size=10)
        
        self.mqtt_pub_client.loop_start()
        self.mqtt_sub_client.loop_start()

        self.ts_start_camera_read = time.perf_counter()
        self.ts_processed_img = time.perf_counter()
        self.ts_pub_mqtt_camera_img = time.perf_counter()

        self.ts_mqtt_segmented_img = time.perf_counter()
        self.ts_processed_segmented_img = time.perf_counter()
        self.ts_ros_pub_segmented_img = time.perf_counter()

        self.benchmarking_values = np.array([[]])


    def callback(self, data):
        try:
            self.ts_start_camera_read = time.perf_counter()
            rospy.loginfo_once("reading from camera feed")

            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            __,jpeg = cv2.imencode('.jpg',cv_image)
            self.ts_processed_img = time.perf_counter()

            #cv2.imshow('vehicle',cv_image)
            #cv2.waitKey(0)
            #plt.show(0)

            self.mqtt_pub_client.publish(MQTT_PUB_CAMERA_TOPIC,jpeg.tobytes())
            self.ts_pub_mqtt_camera_img = time.perf_counter()
            rospy.loginfo_once("img published to broker at topic %s",MQTT_PUB_CAMERA_TOPIC)
        
        except self.bridge.CvBridgeError as e:
            rospy.logerr(e)
    
    def on_mqtt_message(self, client, userdata, msg):
        self.ts_mqtt_segmented_img= time.perf_counter()
        rospy.loginfo("Received segmented image from MQTT")

        np_arr = np.frombuffer(msg.payload, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # Seems to work with Jpg format
        ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        self.ts_processed_segmented_img= time.perf_counter()
        
        # Display image using matplotlib
        #plt.imshow(cv_image)
        #plt.show(1)

        # Publish image to ROS topic
        self.ros_pub.publish(ros_image)
        self.ts_ros_pub_segmented_img= time.perf_counter()

        rospy.loginfo("total_time: %s", self.ts_ros_pub_segmented_img - self.ts_start_camera_read)



    def run(self):
        rospy.loginfo("starting vehicle node")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':

    rospy.init_node('vehicle_node', anonymous=True)
    vehicle_node = VehicleNode()

    try:
        vehicle_node.run()
    except rospy.ROSInterruptException:
        pass


