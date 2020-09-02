
import numpy as np
import pathlib


# global config
DEFAULT_CONFIDENCE = 0.5
DIR_PATH = '/opt/object_detection_app/'
# DIR_PATH = '/Users/dhyungseoklee/Projects/ML_Pipelines/object-detection-app/'
IMAGES = 'images/'
IMAGES_PATH = DIR_PATH + IMAGES

# MobileNet config

MN_CLASSES = [
	"background", 
	"aeroplane", 
	"bicycle", 
	"bird", 
	"boat",
	"bottle", 
	"bus", 
	"car", 
	"cat", 
	"chair", 
	"cow", 
	"diningtable",
	"dog", 
	"horse", 
	"motorbike", 
	"person", 
	"pottedplant", 
	"sheep",
	"sofa", 
	"train", 
	"tvmonitor"
]

MN_COLORS = np.random.uniform(0, 255, size=(len(MN_CLASSES), 3))				


# for Docker image
prototxt = 'model/MobileNetSSD_deploy.prototxt.txt'
mn_model = 'model/MobileNetSSD_deploy.caffemodel'

MN_PROTOTXT_PATH = DIR_PATH + prototxt
MN_MODEL_PATH = DIR_PATH + mn_model



# YOLO config
NMS_THRESHOLD = 0.4

labels_file = 'model/coco.names'
labels_path = DIR_PATH + labels_file
YL_LABELS = open(labels_path).read().strip().split('\n')
YL_COLORS = np.random.uniform(0, 255, size=(len(YL_LABELS), 3))

weights = 'model/yolov3.weights'
yl_config = 'model/yolov3.cfg'

YL_WEIGHTS_PATH = DIR_PATH + weights
YL_CONFIG_PATH = DIR_PATH + yl_config









