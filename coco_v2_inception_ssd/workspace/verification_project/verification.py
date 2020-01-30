import sys
import cv2
import numpy as np
import tensorflow as tf
import os

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

PATH_TO_GRAPH = 'E:/TensorFlow/workspace/training_demo/trained-inference-graph-12117/frozen_inference_graph.pb'
PATH_TO_LABELS = 'E:/TensorFlow/workspace/training_demo/annotations/label_map.pbtxt'
ACCEPT_DIRECTORY = 'E:/TensorFlow/workspace/training_demo/images/accept'
REJECT_DIRECTORY = 'E:/TensorFlow/workspace/training_demo/images/reject'
ACCEPT_IMAGES = os.listdir(ACCEPT_DIRECTORY)
REJECT_IMAGES = os.listdir(REJECT_DIRECTORY)

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

MAX_PIXELS = 23
MIN_REJECT_KNOTS = 8

false_positives = 0
false_negatives = 0
negatives = 0
positives = 0

for image_path in ACCEPT_IMAGES:
    image = cv2.imread(ACCEPT_DIRECTORY + '/' + image_path)
    image_expanded = np.expand_dims(image, axis = 0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict ={image_tensor: image_expanded})

    indexes = []
    for i in range(classes.size):
        if (classes[0][i] in range(1, 91) and scores[0][i] > 0.5):
            indexes.append(i)
    filtered_boxes = boxes[0][indexes, ...]
    filtered_scores = scores[0][indexes, ...]
    filtered_classes = classes[0][indexes, ...]
    filtered_classes = list(set(filtered_classes))
    filtered_classes = [int(i) for i in filtered_classes]

    width, height, channel = image.shape

    biggest_knot_axis_size = 0
    for box in filtered_boxes:
        ymin = box[0] * height
        xmin = box[1] * width
        ymax = box[2] * height
        xmax = box[3] * width
        knot_axis_size = max(ymax-ymin, xmax - xmin)
        if biggest_knot_axis_size < knot_axis_size:
            biggest_knot_axis_size = knot_axis_size
    if biggest_knot_axis_size > MAX_PIXELS:
        false_positives += 1
    else:
        positives += 1

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates = True,
        line_thickness = 2,
        min_score_thresh = 0.5)

    taco = [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5]
    #print(str(len(taco)))


for image_path in REJECT_IMAGES:
    image = cv2.imread(REJECT_DIRECTORY + '/' + image_path)
    image_expanded = np.expand_dims(image, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    indexes = []
    for i in range(classes.size):
        if (classes[0][i] in range(1, 91) and scores[0][i] > 0.5):
            indexes.append(i)

    filtered_boxes = boxes[0][indexes, ...]
    filtered_scores = scores[0][indexes, ...]
    filtered_classes = classes[0][indexes, ...]
    filtered_classes = list(set(filtered_classes))
    filtered_classes = [int(i) for i in filtered_classes]

    width, height, channel = image.shape

    biggest_knot_axis_size = 0
    for box in filtered_boxes:
        ymin = box[0] * height
        xmin = box[1] * width
        ymax = box[2] * height
        xmax = box[3] * width
        knot_axis_size = max(ymax - ymin, xmax - xmin)
        if biggest_knot_axis_size < knot_axis_size:
            biggest_knot_axis_size = knot_axis_size
    if biggest_knot_axis_size > MAX_PIXELS:
        negatives += 1
    else:
        false_negatives += 1

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
        min_score_thresh=0.5)

    taco = [category_index.get(value) for index, value in enumerate(classes[0]) if scores[0, index] > 0.5]
    #print(str(len(taco)))
#
#    cv2.imshow('Sek Detector', image)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#
#
#
print("Positives: " + str(positives))
print("Negatives: " + str(negatives))
print("False positives: " + str(false_positives))
print("False negatives: " + str(false_negatives))