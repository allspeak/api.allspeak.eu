FROM python:3.6.1

RUN cd /var

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Create the group and user to be used in this container
RUN groupadd flaskgroup && useradd -m -g flaskgroup -s /bin/bash flask

RUN mkdir /home/flask/app

RUN chown -R flask:flaskgroup /home/flask/app

USER flask