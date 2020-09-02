import streamlit as st
from PIL import Image
import numpy as np
import cv2

import config
from config import YL_COLORS

# get layer outputs by passing image into yolo algorithm
@st.cache
def yl_detect_image(image):

	blob = cv2.dnn.blobFromImage(image, 1 / 225.0, (416,416), swapRB = True, crop = False)

	net = cv2.dnn.readNetFromDarknet(config.YL_CONFIG_PATH, config.YL_WEIGHTS_PATH)
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	net.setInput(blob)
	layerOutputs = net.forward(ln)

	return layerOutputs


# extract metrics from each layer of layer outputs
@st.cache
def yl_get_metrics(image,layersOutputs, confidence_threshold = config.DEFAULT_CONFIDENCE):

	boxes = []
	confidences = []
	classIDs = []

	(h,w) = image.shape[:2]

	for output in layersOutputs:
		for detection in output:

			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > confidence_threshold:
				# Yolo spits out center (x,y) of boxes
				# followed by width and height
				box = detection[0:4] * np.array([w,h,w,h])
				(centerX, centerY, width, height) = box.astype('int')

				x = int(centerX - (width/2))
				y = int(centerY - (height/2))

				boxes.append([x,y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	return boxes, confidences, classIDs


# draw boxes and labels with extracted output
@st.cache
def yl_draw_boxes(image, boxes, confidences, classIDs, threshold = config.NMS_THRESHOLD, confidence = config.DEFAULT_CONFIDENCE):

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)
	labels = []

	if len(idxs) >= 0:
		for i in idxs.flatten():
			(x,y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			color = [int(c) for c in YL_COLORS[classIDs[i]]]
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = f"{config.YL_LABELS[classIDs[i]]}: {round(confidences[i]*100, 2)}%"
			labels.append(text)
			cv2.putText(image,text, (x,y -5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	return image, labels
