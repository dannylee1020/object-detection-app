import streamlit as st
from PIL import Image
import numpy as np
import cv2

import config
from config import MN_CLASSES, MN_COLORS


@st.cache
def mn_detect_image(image):

	blob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300,300)), 0.007483, (300,300), 127.5
	)

	net = cv2.dnn.readNetFromCaffe(config.MN_PROTOTXT_PATH, config.MN_MODEL_PATH)
	net.setInput(blob)
	detections = net.forward()

	return detections

@st.cache
def mn_draw_boxes(image, detections, confidence_threshold = config.DEFAULT_CONFIDENCE):

	(h,w) = image.shape[:2]
	labels = []
	for i in np.arange(0, detections.shape[2]):
		confidence = detections[0,0,i,2]

		if confidence > confidence_threshold:

			idx = int(detections[0,0,i,1])
			box = detections[0, 0, i, 3:7] * np.array([w,h,w,h])
			(startX, startY, endX, endY) = box.astype('int')


			label = f"{MN_CLASSES[idx]}: {round((confidence*100),2)}%"
			labels.append(label)
			cv2.rectangle(image, (startX, startY), (endX, endY), MN_COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY  + 15
			cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MN_COLORS[idx], 2)


	return image, labels