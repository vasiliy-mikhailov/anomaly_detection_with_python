FROM python:3.12-alpine
RUN mkdir src
WORKDIR /src
RUN apk add g++
RUN apk add libffi-dev
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py .
COPY model.joblib .
CMD uvicorn main:app --host 0.0.0.0 --port 8888
# CMD sh
