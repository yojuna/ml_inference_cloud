import paho.mqtt.client as mqtt
import numpy as np
from cv_bridge import CvBridge
import cv2
import argparse
import os

from frozen_graph_runner import thefrozenfunc, thesavedfunc, model_initialiser

# MQTT Broker
MQTT_BROKER_IP = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_SUB_CAMERA_TOPIC = "/vehicle_camera"
MQTT_PUB_SEGMENTED_TOPIC = "/segmented_images"

class CloudNode:
    def __init__(self,model, color_palette,use_saved_model):
        self.bridge = CvBridge()
        
        # Model Initialisation
        self.model = model
        self.color_palette = color_palette
        self.use_saved_model = use_saved_model


        #setup MQTT Client
        self.mqtt_pub_client = mqtt.Client() # Publish segmented images to vehicle node
        self.mqtt_sub_client = mqtt.Client() # Subscribe camera images from vehicle node
        self.mqtt_pub_client.connect(MQTT_BROKER_IP,MQTT_BROKER_PORT,61) #same client as the segmented image subscribing on vehicle node
        self.mqtt_sub_client.connect(MQTT_BROKER_IP,MQTT_BROKER_PORT,60) #same client as the camera image publishing on vehicle node
        self.mqtt_sub_client.on_message = self.on_mqtt_message

        # List to vehicle's camera (ROS SUB)
        #self.ros_sub = rospy.Subscriber(ROS_SUB_CAMERA_TOPIC,Image,self.callback)
        #Callback has the MQTT sending to cloud part

        # Receive data from cloud through MQTT topic
        self.mqtt_sub_client.subscribe(MQTT_SUB_CAMERA_TOPIC)

        self.mqtt_pub_client.loop_start()
        self.mqtt_sub_client.loop_start()

    def on_mqtt_message(self, client, userdata, msg):
        print("Received vehicle image from MQTT")
        np_arr = np.frombuffer(msg.payload, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Inference function
        if self.use_saved_model:
            inferred_image = thesavedfunc(cv_image,self.model,self.color_palette)

        else:
            inferred_image = thefrozenfunc(cv_image,self.model,self.color_palette)

        # Publishing inferred image through MQTT Topic
        _, jpeg = cv2.imencode('.jpg', inferred_image)
        self.mqtt_pub_client.publish(MQTT_PUB_SEGMENTED_TOPIC, jpeg.tobytes())
        print("Image published to broker at topic %s", MQTT_PUB_SEGMENTED_TOPIC)

    def run(self):
        print("Starting cloud node")
        while True:
            pass

if __name__ == '__main__':
    current_directory = os.getcwd()
    print("Current working directory:", current_directory)

    parser = argparse.ArgumentParser(description="Segmentation Script with options")

    model_help = """Path to the model file (frozen graph or SavedModel).
                    Runs mobilenetv3_large_os8_deeplabv3plus_72miou model (for saved_model) and 
                    mobilenet_v3_small_968_608_os8.pb (for frozen_model) present inside model folder as DEFAULT.
                    Add your segmentation model directly into the model folder and put the path relative starting with model/...
                    """

    xml_help = """Path to the xml file containing segmentation labels.
                   Runs cityscapes.xml (for saved_model) and convert.xml (for frozen_model) as DEFAULT.
                   Add your segmentation label file directly into the xml folder and put the path relative starting with xml/...
                   """

    graph_help = """ Use the flag to use the Saved models instead of Frozen model.
                     Don't forget to use the corresponding model type for model_path. Will resort to DEFAULT if any error
                 """

    parser.add_argument("-model_path", nargs='?', default='model/mobilenet_v3_small_968_608_os8.pb', help=model_help)
    parser.add_argument("-xml_path", nargs='?', default='xml/convert.xml', help=xml_help)
    parser.add_argument("--use_saved_model", action="store_true", help=graph_help)
    args = parser.parse_args()
    
    if args.use_saved_model:
        if not os.path.isdir(args.model_path):
            print("Model is not a savedModel, switching to Default Saved Model here")
            args.model_path = None
            args.xml_path = 'xml/cityscapes.xml'
    else:
        if not os.path.isfile(args.model_path):
            print("Model given is not a Frozen graph, could be a SavedModel or just a dir, switching to default Frozen graph model")
            args.model_path = None

    model,color_palette = model_initialiser(model_path=args.model_path, xml_path=args.xml_path, use_saved_model=args.use_saved_model)
    cloud_node = CloudNode(model,color_palette,args.use_saved_model)
    try:
        cloud_node.run()
    except KeyboardInterrupt:
        pass
