#!/usr/bin/python3


import rospy
from sensor_msgs.msg import Image
import paho.mqtt.client as mqtt
import numpy as np
from cv_bridge import CvBridge
import cv2
import argparse
import os
import sys

import matplotlib as plt

# for running the models
from frozen_graph_runner import thefrozenfunc, thesavedfunc, model_initialiser


ros_args = rospy.myargv(argv=sys.argv)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# to use CPU instead of GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# MQTT Broker
MQTT_BROKER_IP = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_SSL_BROKER_PORT = 8883

MQTT_SUB_CAMERA_TOPIC = "/vehicle_camera"
MQTT_PUB_SEGMENTED_TOPIC = "/segmented_images"

MQTT_CERTS_DIR = "/home/rosuser/ws/ws_mqtt_nodes/cloud_node/src/mqtt_node/src/"

MQTT_SSL_CLIENT_CERT = MQTT_CERTS_DIR + "cert/client/client-cert.pem"
MQTT_SSL_CLIENT_KEY = MQTT_CERTS_DIR + "cert/client/client-key.pem"
MQTT_SSL_CLIENT_CSR =  MQTT_CERTS_DIR + "cert/client/client-csr.pem"
MQTT_SSL_CLIENT_CA =  MQTT_CERTS_DIR + "cert/ca-cert.pem"

# ROS
ROS_PUB_CAMERA_TOPIC = "/cloud_ros_camera"


class CloudNode:
    def __init__(self,model, color_palette,use_saved_model):
        self.bridge = CvBridge()
        self.ssl_enabled = True

        
        # Model Initialisation
        self.model = model
        self.color_palette = color_palette
        self.use_saved_model = use_saved_model


        #setup MQTT Client
        self.mqtt_pub_client = mqtt.Client() # Publish segmented images to vehicle node
        self.mqtt_sub_client = mqtt.Client() # Subscribe camera images from vehicle node


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



        # self.mqtt_pub_client.connect(MQTT_BROKER_IP,MQTT_BROKER_PORT,61) #same client as the segmented image subscribing on vehicle node
        # self.mqtt_sub_client.connect(MQTT_BROKER_IP,MQTT_BROKER_PORT,60) #same client as the camera image publishing on vehicle node
        self.mqtt_sub_client.on_message = self.on_mqtt_message

        # List to vehicle's camera (ROS SUB)
        #self.ros_sub = rospy.Subscriber(ROS_SUB_CAMERA_TOPIC,Image,self.callback)
        #Callback has the MQTT sending to cloud part

        # Receive data from cloud through MQTT topic
        self.mqtt_sub_client.subscribe(MQTT_SUB_CAMERA_TOPIC)
        self.ros_pub = rospy.Publisher(ROS_PUB_CAMERA_TOPIC, Image, queue_size=10)
        
        self.mqtt_pub_client.loop_start()
        self.mqtt_sub_client.loop_start()

        self.benchmarking = np.array()

    # def callback(self,data):
    #     try:
    #         rospy.loginfo_once("reading from camera feed")
    #         cv_image = self.bridge.imgmsg_to_cv2(data,desired_encoding='bgr8')
    #         __,jpeg = cv2.imencode('.jpg',cv_image)
    #         self.mqtt_pub_client.publish(MQTT_PUB_CAMERA_TOPIC,jpeg.tobytes())
    #         rospy.loginfo_once("img published to broker at topic %s",MQTT_PUB_CAMERA_TOPIC)
        
    #     except self.bridge.CvBridgeError as e:
    #         rospy.logerr(e)
    
    def on_mqtt_message(self, client, userdata, msg):
        rospy.loginfo("Received vehicle image from MQTT broker from topic: %s", MQTT_SUB_CAMERA_TOPIC)

        # preprocessing image for inference
        np_arr = np.frombuffer(msg.payload, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # Seems to work with Jpg format
        
        # Display image using matplotlib
        #plt.imshow(cv_image)
        #plt.show(1)

        # # Publish image to ROS topic
        # # publishing vehicle image as a ROS Topic incase you got 
        # # other operations to do on the cloud
        # ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        # rospy.loginfo("processed vehicle image from MQTT broker")

        # self.ros_pub.publish(ros_image)
        # rospy.loginfo("processed vehicle camera image and published to ROS for inference:")
        
        # # Inference function 
        # starting inference
        rospy.loginfo("starting inference")
        if self.use_saved_model:
            # print("running saved model")
            inferred_image = thesavedfunc(cv_image, self.model, self.color_palette)
            rospy.loginfo("saved Func inference")
        else:
            # print("running frozen func model")
            inferred_image = thefrozenfunc(cv_image, self.model, self.color_palette)
            rospy.loginfo("frozen Func inference")

        rospy.loginfo("inference completed")
        
        #inferred_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        #inferred_image = thesavedfunc(cv_image) # To infer from models in saved model format
        # inferred_image = thefrozenfunc(cv_image) # To infer from models in frozen graph state

        # Publishing infered image through MQTT Topic
        __,jpeg = cv2.imencode('.jpg',inferred_image)
        self.mqtt_pub_client.publish(MQTT_PUB_SEGMENTED_TOPIC, jpeg.tobytes())
        rospy.loginfo("img published to broker at topic %s",MQTT_PUB_SEGMENTED_TOPIC)

    def run(self):
        rospy.loginfo("starting cloud node")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':

    #image_subscriber = ImageSubscriber()
    #image_publisher=ImagePublisher()


    current_directory = os.getcwd()
    print("Current working directory:", current_directory)
    
    parser = argparse.ArgumentParser(description="Segmentation Script with options")
    
    model_help = """Path to the model file (frozen graph or SavedModel).
                    Runs mobilenetv3_large_os8_deeplabv3plus_72miou model (for saved_model) and 
                    mobilenet_v3_small_968_608_os8.pb (for frozen_model)present inside model folder as DEFAULT.
                    Add your segmentation model directly into the model folder and put the path relative starting with model/...       
                    """
    
    xml_help =  """Path to the xml file containing segmentation labels.
                   Runs cityscapes.xml (for saved_model) and convert.xml (for frozen_model) as DEFAULT.
                   Add your segmentation label file directly into the xml folder and put the path relative starting with xml/...
                   """
    
    graph_help = """ Use the flag to use the Saved models instead of Frozen model.
                     Don't forget to use the corresponding model type for model_path. Will resort to DEFAULT if any error
                 """
    
    parser.add_argument("-m","--model_path", default='model/mobilenet_v3_small_968_608_os8.pb',help=model_help)
    parser.add_argument("-x","--xml_path", default='xml/convert.xml',help=xml_help)
    parser.add_argument("-s","--use_saved_model", action="store_true", help=graph_help)

    args = parser.parse_args(ros_args[1:])
    print("Model Path: ", args.model_path)
    print("XML Path: ", args.xml_path)
    # print("")

    if args.use_saved_model:
        if not os.path.isdir(args.model_path):
            print("Model is not a savedModel, switching to Default Saved Model here")
            args.model_path = None
            args.xml_path = 'xml/cityscapes.xml' # Explicitly defining, else takes the convert.xml

    else:
        if not os.path.isfile(args.model_path):
            print("Model given is not a Frozen graph, could be a SavedModel or just a dir, switching to default Frozen graph model")
            args.model_path = None

    rospy.init_node('vehicle_node',anonymous=True)
    print("Running the segmentation model with {} model and {} label file".format(args.model_path,args.xml_path))
    
    model,color_palette = model_initialiser(model_path=args.model_path, xml_path=args.xml_path, use_saved_model=args.use_saved_model)
    cloud_node = CloudNode(model,color_palette,args.use_saved_model)
    try:
        cloud_node.run()
    except rospy.ROSInterruptException:
        pass
