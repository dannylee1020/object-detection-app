FROM python:3.7

# define working directory within docker 
WORKDIR /opt/object_detection_app

# copy and install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt


# copy necessary folders
COPY app /opt/object_detection_app/app/
COPY images /opt/object_detection_app/images/
COPY model /opt/object_detection_app/model/

# for local build 
EXPOSE 8501

# for local testing
ENTRYPOINT ["streamlit", "run"]
CMD ["app/app.py"]


# for Heroku Deployment
#CMD streamlit run app/app.py --server.port $PORT