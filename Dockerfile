FROM python:3.12.3
COPY . /TumorDetectApp
WORKDIR /TumorDetectApp
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:TumorDetectApp