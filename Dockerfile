# https://github.com/entmike/stylegan2

FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN apt-get update && \
    apt-get install cmake git unzip zip wget -y 

RUN pip install scipy==1.3.3
RUN pip install requests==2.22.0
RUN pip install Pillow==6.2.1

# Install Addl Deps
RUN pip install IPython && \
    pip install imageio && \
    pip install keras==2.3.1 && \
    pip install dlib

RUN mkdir /models && cd /models && \
    wget http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl && \
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
    wget http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/vgg16_zhang_perceptual.pkl

RUN cd / && \
    mkdir /in /out /latents /records && \
    git clone https://github.com/entmike/stylegan2 && \
    cd /stylegan2

# Sample Files
COPY ./samples/ /in/