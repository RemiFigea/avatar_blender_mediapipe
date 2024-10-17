FROM python:3.12-slim

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install ffmpeg libsm6 libxext6 wget xz-utils libegl1-mesa libgl1-mesa-glx -y && \
    apt-get clean

# Install the dependancies
COPY requirements.txt /
RUN pip install --no-cache-dir -r requirements.txt

# Install Blender
RUN wget https://download.blender.org/release/Blender4.2/blender-4.2.2-linux-x64.tar.xz && \
    tar -xvf blender-4.2.2-linux-x64.tar.xz && \
    mv blender-4.2.2-linux-x64 /opt/blender && \
    rm blender-4.2.2-linux-x64.tar.xz

# Install the dependancies in Blender environnement
RUN /opt/blender/4.2/python/bin/python3.11 -m pip install -r /requirements.txt

# Copy the code files
COPY src /

# Listen to port 5000 (the default port of flask)
EXPOSE 5000

# Define the working dir in the contener
WORKDIR /

# Command to start the app
CMD ["python", "app.py"]
