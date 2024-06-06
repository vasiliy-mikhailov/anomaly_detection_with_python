FROM python:3.12-alpine
RUN mkdir src
WORKDIR /src
RUN apk add g++
RUN apk add libffi-dev
RUN pip install jupyter
CMD jupyter notebook --no-browser --allow-root --ip 0.0.0.0 --port 8888
