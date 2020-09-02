# Object Detection with YOLO and MobileNet SSD

## Project Overview
Object Detection app created with Streamlit. Since training objection detection algorithm is very expensive in terms of both computation and time, so using pre-trained model in this project. Adjust the confidence threshold of the model to see how model's detections change. Since MobileNet was trained on fewer class targets and has smaller model architecture, YOLO will be able to detect more objects with higher accuracy. 

## Model Overview
Single Shot Detectors (SSD) is one of many popular object detection methods along with YOLO and R-CNN. SSD is generally known to be positioned between R-CNN and YOLO in terms of both accuracy and speed. It's a lot faster than R-CNN and usually more accurate than YOLO algorithm. Combined with MobileNet, which was developed by Google for resource constrained devices such as smartphones, it is capable of producing even faster computation while preserving decent accuracy. 
<br>
<br>
You Only Look Once (YOLO) algorithm is also widely used algorithm in object detection. It's known for its speed and achieving good accuracy, which is why YOLO is popular in real-time object detection problems. 

## Run with Docker
From the root directory: 

	docker build -t dannylee1020/object_detection .
	docker run -p 8501:8501 dannylee1020/object_detection:latest

Then visit [localhost:8501](https://localhost:8501) to see streamlit app.

## Reference
[MobileNet SSD](https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/) by Adrian Rosebrock
<br>
[YOLO](https://pjreddie.com/darknet/yolo/)
<br>
[Streamlit](https://www.streamlit.io/)