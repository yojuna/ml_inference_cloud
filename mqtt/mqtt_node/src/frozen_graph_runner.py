import tensorflow as tf
from utils.img_utils import resize_image
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import time
import rospy

# benchmarking_table = np.array([])

def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


def load_frozen_graph(path_to_frozen_graph):
    sess = None
    graph = tf.Graph()

    input_tensor_name = 'input:0'
    output_tensor_name = 'prediction:0'

    with tf.io.gfile.GFile(path_to_frozen_graph, 'rb') as file_handle:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(file_handle.read())

    # Wrap frozen graph to ConcreteFunctions
    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                                inputs=[input_tensor_name],
                                                outputs=[output_tensor_name],
                                                print_graph=True)
    return frozen_func

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

    ### START CODE HERE ###
    
    # Task 1:
    # Replace the following command
    rgb_encoding = color_palette[segmentation_map]

    ### END CODE HERE ###
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

def model_initialiser(model_path, xml_path, use_saved_model):
    if use_saved_model:
        if model_path == None:
            model_path='model/mobilenetv3_large_os8_deeplabv3plus_72miou'
        model = tf.saved_model.load(model_path)
    
    else:
        if model_path == None:
            model_path = 'model/mobilenet_v3_small_968_608_os8.pb'
        model = load_frozen_graph(model_path)
    color_palette,__,__ = parse_convert_xml(xml_path)
    return model,color_palette

def thefrozenfunc(input_img, model, color_palette):
    start = time.perf_counter()

    width = 968 # Frozen model's input is 968*608
    height = 608

    preprocess_start = time.perf_counter()
    input_img = resize_image(input_img,[height,width])
    #input_img = input_img / 255.0 #normalisation not required for frozen graphs
    input_img = input_img[None]


    prediction_start = time.perf_counter()
    predictions = model(tf.cast(input_img,tf.uint8))
    prediction_end = time.perf_counter()

    prediction = tf.squeeze(predictions).numpy()
    prediction = segmentation_map_to_rgb(prediction,color_palette).astype(np.uint8)
    prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)
    end = time.perf_counter()

    # print("Entire time: %s", end - start)
    # print("Prediction time: %s", prediction_end - prediction_start)
    # print("Postprocessing: %s", end - prediction_end)

    total_time = end - start
    prediction_time = prediction_end - prediction_start
    preprocess_step = prediction_start - preprocess_start
    postprocess_step = end - prediction_end


    # benchmarking_table.append(np.array([total_time, prediction_time, preprocess_step, postprocess_step]))

    rospy.loginfo("Entire time: %s", total_time)
    rospy.loginfo("Prediction time: %s", prediction_time)
    rospy.loginfo("Preprocessing step: %s", preprocess_step)
    rospy.loginfo("Postprocessing: %s", postprocess_step)

    # rospy.loginfo("average_total_time: %s", np.mean(benchmarking_table, axis=0)[0])

    return prediction

def thesavedfunc(input_img, model, color_palette):
    # Reads from savedModel

    start = time.perf_counter()
    width = 2048 # Saved model's input is 2048*1024
    height = 1024

    #input_img = cv2.imread(image)
    #input_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)
    preprocess_start = time.perf_counter()
    input_img = resize_image(input_img, [height, width])
    input_img = input_img / 255.0 #normalisation
    input_img = np.expand_dims(input_img, axis=0)
    input_img = tf.cast(input_img, dtype=tf.float32)

    prediction_start = time.perf_counter()
    predictions = model(input_img)
    prediction_end = time.perf_counter()

    prediction = tf.squeeze(predictions).numpy()
    argmax_prediction = np.argmax(prediction, axis=2)
    prediction = segmentation_map_to_rgb(argmax_prediction, color_palette).astype(np.uint8)
    prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)
    end = time.perf_counter()

    # print("Entire time", end - start)
    # print("Prediction time", prediction_end - prediction_start)
    # print("Preprocessing step", prediction_start - preprocess_start)
    # print("Postprocessing", end - prediction_end)

    total_time = end - start
    prediction_time = prediction_end - prediction_start
    preprocess_step = prediction_start - preprocess_start
    postprocess_step = end - prediction_end

    # benchmarking_table.append(np.array([total_time, prediction_time, preprocess_step, postprocess_step]))

    rospy.loginfo("Entire time: %s", total_time)
    rospy.loginfo("Prediction time: %s", prediction_time)
    rospy.loginfo("Preprocessing step: %s", preprocess_step)
    rospy.loginfo("Postprocessing: %s", postprocess_step)

    # rospy.loginfo("average_total_time: %s", np.mean(benchmarking_table, axis=0)[0])

    
    return prediction
# Setup 

# if __name__ == "__main__":
#     path_image = 'data/image.png'
#     model, color_palette = model_initialiser('model/best_weights_e=00231_val_loss=0.1518','xml/cityscapes.xml',use_saved_model=True)
#     image = cv2.imread(path_image)
#     prediction = thesavedfunc(image,model,color_palette)
#     #prediction = thefrozenfunc(image,model,color_palette)
#     #cv2.imshow('prediction',prediction)
#     #cv2.waitKey(0)
