import json
import requests
import numpy as np
from img_utils import resize_image
import cv2
import xml.etree.ElementTree as ET
import time
import speedtest
from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr
import tensorflow as tf
import argparse
# Google Remote Procedure Call
import grpc
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc



def segmentation_map_to_rgb(segmentation_map,color_palette):
    """
    Converts segmentation map to a RGB encoding according to self.color_palette
    Eg. 0 (Class 0) -> Pixel value [128, 64, 128] which is on index 0 of self.color_palette
        1 (Class 1) -> Pixel value [244, 35, 232] which is on index 1 of self.color_palette

    self.color_palette has shape [256, 3]. Each index of the first dimension is associated
    with an RGB value. The index corresponds to the class ID.

    :param segmentation_map: ndarray numpy with shape (height, width)
    :return: RGB encoding with shape (height, width, 3)
    """
    rgb_encoding = color_palette[segmentation_map]
    return rgb_encoding

def parse_convert_xml(conversion_file_path):
    """
    Parse XML conversion file and compute color_palette 
    """

    defRoot = ET.parse(conversion_file_path).getroot()

    color_to_label = {}

    color_palette = np.zeros((256, 3), dtype=np.uint8)
    class_list = np.ones((256), dtype=np.uint8) * 255
    class_names = np.array(["" for _ in range(256)], dtype='<U25')
    for idx, defElement in enumerate(defRoot.findall("SLabel")):
        from_color = np.fromstring(defElement.get("fromColour"), dtype=int, sep=" ")
        to_class = np.fromstring(defElement.get("toValue"), dtype=int, sep=" ")
        class_name = defElement.get('Name').lower()
        if to_class in class_list:
            color_to_label[tuple(from_color)] = int(to_class)
        else:
            color_palette[idx] = from_color
            class_list[idx] = to_class
            class_names[idx] = class_name
            color_to_label[tuple(from_color)] = int(to_class)

    # Sort classes accoring to is train ID
    sort_indexes = np.argsort(class_list)

    class_list = class_list[sort_indexes]
    class_names = class_names[sort_indexes]
    color_palette = color_palette[sort_indexes]

    return color_palette, class_names, color_to_label

def predict_rest(json_data, url,color_palette):
    """
    Make predictions using a remote machine learning model served through a REST API.

    Parameters:
    - json_data: JSON-formatted data containing input for the model.
    - url: The URL endpoint of the TensorFlow Serving server hosting the model.
    - color_palette: A color palette used for post-processing the model's predictions.

    Returns:
    - prediction: The processed prediction result, typically an image or classification output.

    Description:
    This function sends a JSON-formatted input to a remote TensorFlow Serving server specified by the 'url'.
    The server processes the input using the hosted model and returns a prediction response in JSON format.
    The prediction response is then post-processed to obtain the final prediction, often an image or classification result,
    which is returned as the output of this function.

    Note:
    - The function assumes that the TensorFlow Serving server is correctly set up to handle REST API requests.
    - The 'color_palette' parameter is used for converting the model's segmentation map into a colored image.

    Models Available:
    -  best_weights_e=00231_val_loss=0.1518
    -  mobilenetv3_large_os8_deeplabv3plus_72miou

    Example Usage:
    - prediction = predict_rest(json_input, "http://example.com/model_endpoint", color_palette)
    """
    prediction_start = time.time()
    json_response = requests.post(url, data=json_data)
    response = json.loads(json_response.text)
    prediction_end = time.time()
    prediction_time = time.time()-prediction_start
    #print(response)
    predictions = np.array(response["predictions"])
    #print(predictions.shape)
    # prediction = tf.squeeze(predictions).numpy()
    prediction = np.squeeze(predictions).tolist()  # Convert to list
    argmax_prediction = np.argmax(prediction, axis=2)
    prediction = segmentation_map_to_rgb(argmax_prediction,color_palette=color_palette).astype(np.uint8)
    postprocessing_time = time.time() - prediction_end
    #prediction = cv2.cvtColor(prediction,cv2.COLOR_BGR2RGB)

    return prediction, prediction_time,postprocessing_time

def predict_grpc(data, input_name, stub,color_palette):
    """
    Make predictions using gRPC for a machine learning model served by TensorFlow Serving.

    Parameters:
    - data: Input data for prediction.
    - input_name: The name of the input tensor in the model.
    - stub: gRPC stub for communicating with the TensorFlow Serving server.
    - color_palette: A color palette used for post-processing the prediction.

    Returns:
    - prediction: The processed prediction result.

    Description:
    This function sends input data to a TensorFlow Serving server using gRPC for model prediction.
    It assumes the server is hosting a semantic segmentation model.
    The function processes the gRPC response, converts it to an image format, and applies color mapping using the provided color_palette.
    """
    grpc_request_preprocess = time.time()

    # Create a gRPC request made for prediction
    request = predict_pb2.PredictRequest()

    # Set the name of the model, for this use case it is "model"
    request.model_spec.name = "mobilenet" # Based on the Tensorflow Docker command, under the MODEL_NAME

    # Set which signature is used to format the gRPC query
    # here the default one "serving_default"
    request.model_spec.signature_name = "serving_default"

    # Set the input as the data
    # tf.make_tensor_proto turns a TensorFlow tensor into a Protobuf tensor
    request.inputs[input_name].CopyFrom(tf.make_tensor_proto(data.tolist()))
    preprocess_end = time.time()
    preprocessing_time = preprocess_end - grpc_request_preprocess
    # Send the gRPC request to the TF Server
    result = stub.Predict(request)
    prediction_end = time.time()
    prediction_time = prediction_end-preprocess_end
    # Process the gRPC response
    output_name = list(result.outputs.keys())[0]
    output_data = result.outputs[output_name].float_val  # Assuming the output is in float format

    # Convert the float data to an image format
    output_data = np.array(output_data)
    height = 1024
    width = 2048
    num_channels = 20
    output_data = output_data.reshape((1, height, width, num_channels))  # Adjust the shape accordingly
    #output_data = (output_data * 255).astype(np.uint8)  # Assuming output is in the range [0, 1]

    # Convert the image to RGB

    argmax_prediction = np.argmax(output_data, axis=3)
    prediction = segmentation_map_to_rgb(argmax_prediction, color_palette=color_palette).astype(np.uint8)
    prediction = np.squeeze(prediction)
    postprocessing_time = time.time() - prediction_end
    return prediction, preprocessing_time,prediction_time,postprocessing_time


def bag_reader(bag_file,flag,color_palette):

    '''
    Reads frames from a ROSBag file and processes them using OpenCV and TensorFlow Serving based on the specified flag.

    Args:
        bag_file (str): The path to the ROSBag file containing camera frames.
        flag (str): A flag indicating whether to use TensorFlow Serving ('grpc') or a REST API ('rest') for inference.
    '''

    while True: # Useful to loop the ROSBag as there ain't existing built-in func
        image_retrieve_list, image_preprocess_list, request_preprocess_list, prediction_list,postprocessing_list = []

        with Reader(bag_file) as reader:
        # Iterate over messages
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == '/sensors/camera/left/image_raw':
                    # Assuming 'sensor_msgs/Image' message type
                    image_retrieval_start = time.time()
                    msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
                    
                    # Convert ROS image data to OpenCV image
                    img_data = msg.data
                    width = msg.width
                    height = msg.height
                    encoding = msg.encoding
                    np_arr = np.frombuffer(img_data, np.uint8)
                    reshaped_array = np_arr.reshape((height, width))  # Assuming 3 channels (BGR in OpenCV)
                    reshaped_array = ((reshaped_array - reshaped_array.min()) / (reshaped_array.max() - reshaped_array.min()) * 255).astype(np.uint8)
                    demosaiced_image = cv2.cvtColor(reshaped_array, cv2.COLOR_BAYER_RG2BGR) # Required to convert BAYER format to BGR
                    rgb_image = cv2.cvtColor(demosaiced_image, cv2.COLOR_BGR2RGB) #BGR to RGB
                    image_retrieval_end = time.time()
                    image_retrieve = image_retrieval_end - image_retrieval_start # Time it takes for one frame from the ROSBag
                    
                    image_preprocess,request_preprocess,prediction_time,postprocessing_time = serving_func(rgb_image,flag,color_palette)
                    
                    image_retrieve_list.append(image_retrieve)
                    image_preprocess_list.append(image_preprocess)
                    request_preprocess_list.append(request_preprocess)
                    prediction_list.append(prediction_time)
                    postprocessing_list.append(postprocessing_time)

                    if (len(prediction_list)>5):
                        image_retrieve_list.pop(0)
                        image_preprocess_list.pop(0)
                        request_preprocess_list.pop(0)
                        prediction_list.pop(0)
                        postprocessing_list.pop(0)
                        number = len(image_retrieve_list)
                        avg_image_retrieve = sum(image_retrieve_list)/number
                        avg_image_preprocess = sum(image_preprocess_list)/number
                        avg_request_preprocess = sum(request_preprocess_list)/number
                        avg_prediction = sum(prediction_list)/number
                        avg_postprocess = sum(postprocessing_list)/number
                        
                        print("Average Image Retrieval: ",avg_image_retrieve,"Average Image Preprocessing: ",avg_image_preprocess)
                        print("\n")
                        print("Average Request Preprocessing: ",avg_request_preprocess,"Average Prediction Time: ",avg_prediction)
                        print("\n")
                        print("Average Postprocessing Time: ",avg_postprocess)
                        break
                    
                    
def serving_func(input_img,flag,color_palette):
    '''
    Processes an input image using gRPC or a REST API based on the specified flag and performs various measurements.

    Args:
        input_img (numpy.ndarray): The input image for inference.
        flag (str): A flag indicating whether to use gRPC ('grpc') or a REST API ('rest') for inference.
    '''

    width = 2048
    height = 1024

    # input_img = cv2.imread('image.png')
    preprocess_start = time.time()
    input_img = resize_image(input_img,[height,width])
    input_img = input_img / 255.0



    batched_img = np.expand_dims(input_img, axis=0)
    batched_img = batched_img.astype(float)
    #batched_img = batched_img.astype(np.uint8)
    print(f"Batched image shape: {batched_img.shape}")
    preprocess_end = time.time()
    image_preprocess = preprocess_end-preprocess_start
    REST_request_preprocess = "None"

    if flag == 'rest':
        ### Serving part 
        rest_start = time.time()
        data = json.dumps(
            {"signature_name": "serving_default", "instances": batched_img.tolist()}
        )
        rest_end = time.time()

        # Docker command for setting up Server :
        '''
        CHECK README
        '''
        url = "http://localhost:8501/v1/models/mobilenet:predict" #If tfserving on your local system
        #url = "http://i2200049.ika.rwth-aachen.de:8501/v1/models/mobilenet:predict" # If tfserving on the IKA Workstation

    if flag == 'rest':
        prediction,prediction_time,postprocessing_time = predict_rest(data, url,color_palette)
        request_preprocess = rest_end-rest_start
    else:
        prediction,request_preprocess,prediction_time,postprocessing_time = predict_grpc(batched_img,input_name=input_name,stub=stub,color_palette=color_palette)
    prediction = cv2.cvtColor(prediction,cv2.COLOR_BGR2RGB)

    return image_preprocess,request_preprocess,prediction_time,postprocessing_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify trigger and model_export_path.')
    parser.add_argument('--bag', type=str, default='left_camera_templergraben.bag', help='Path to the Bag file')
    parser.add_argument('--model_export_path', type=str, default='./model/mobilenetv3_large_os8_deeplabv3plus_72miou/', 
                        help='Path to the model export directory. Make sure the model path matches the one TFServing is serving :)')
    parser.add_argument('--trigger', type=str, default='grpc', help='Trigger for mode (e.g., "grpc" or "rest").')
    args = parser.parse_args()
    print("Running the %s Model",args.model_export_path)
    if args.trigger != 'rest': # Anything other than REST API, will trigger the gRPC
        print("Running gRPC mode")
        #gRPC Setting up 
        channel_opt = [('grpc.max_send_message_length', 512 * 1024 * 1024), ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
        channel = grpc.insecure_channel("0.0.0.0:8500", options=channel_opt) #Change this if using any other system other LocalHost for Cloud
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel) # Used to send the gRPC request to the TF Server

        # Get the serving_input key

        model_export_path = args.model_export_path
        loaded_model = tf.saved_model.load(model_export_path)
        input_name = list(
            loaded_model.signatures["serving_default"].structured_input_signature[1].keys()
        )[0]
    else:
        print("Running REST API mode")

    # Bag file 
    path_to_xml = 'xml/cityscapes.xml'

    #path_to_xml = 'convert.xml'
    color_palette, class_names, color_to_label = parse_convert_xml(path_to_xml)
    bag_reader(bag_file=args.bag,flag=args.trigger,color_palette=color_palette)
    

