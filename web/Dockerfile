FROM python:3.6.1

# Create the group and user to be used in this container
RUN groupadd flaskgroup && useradd -m -g flaskgroup -s /bin/bash flask

RUN ls /home/flask

RUN chown -R flask:flaskgroup /home/flask/app/web

USER flask

# Install the package dependencies (this step is separated
# from copying all the source code to avoid having to
# re-install all python packages defined in requirements.txt
# whenever any source code change is made)
RUN pip install --no-cache-dir -r requirements.txt
