import streamlit as st
import os
from PIL import Image
import numpy as np

from mn_ssd import mn_detect_image, mn_draw_boxes
from yolo import yl_detect_image, yl_draw_boxes, yl_get_metrics
import config



def run():

	st.title("Object Detection Algorithm")
	option = st.radio('', ['Choose a test image', 'Upload your own image'])
	st.sidebar.title('Parameters')
	detection_method = st.sidebar.selectbox('Which algorithm do you want to use?', ('YOLO','MobileNet SSD'))

	if detection_method == 'YOLO':
		threshold_slider = st.sidebar.slider("Non-Max Suppression Threshold", 0.0, 1.0, config.NMS_THRESHOLD, 0.05)
	else:
		pass

	confidence_slider = st.sidebar.slider("Confidence threshold for detection", 0.0, 1.0, config.DEFAULT_CONFIDENCE, 0.05)


	if option == 'Choose a test image':
		test_images = os.listdir(config.IMAGES_PATH)
		test_image = st.selectbox('Please choose a test image', test_images)

		# confidence_threshold = st.slider('Confidence threshold for detection', 0.0, 1.0, config.DEFAULT_CONFIDENCE, 0.05)
		show_image = Image.open(config.IMAGES_PATH + test_image)
		st.image(show_image, use_column_width  = True)
		image = np.array(show_image)

	else:
		file_uploaded = st.file_uploader("Upload an image", type = ['png', 'jpg','jpeg'])
		# confidence_threshold = st.slider('Confidence threshold for detection', 0.0, 1.0, config.DEFAULT_CONFIDENCE, 0.05)

		if file_uploaded is not None:
			show_image = Image.open(file_uploaded)
			st.image(show_image, use_column_width = True)
			image = np.array(show_image)

		else:
			st.write('** Please upload an image **')



	if st.button('Detect Objects'):
		
		if detection_method == 'YOLO':
			layer_outputs = yl_detect_image(image)
			boxes, confidences, classIDs = yl_get_metrics(image,layer_outputs, config.DEFAULT_CONFIDENCE)
			image, labels = yl_draw_boxes(image, boxes, confidences, classIDs, threshold_slider, confidence_slider)

		else:
			detections = mn_detect_image(image)
			image, labels = mn_draw_boxes(image, detections, confidence_slider)

		st.image(image, caption = 'Processed Image', use_column_width = True)
		st.write(labels)
 


if __name__ == '__main__':

	run()



