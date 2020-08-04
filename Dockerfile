# https://github.com/entmike/stylegan2

FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN apt-get update && \
    apt-get install cmake git unzip zip -y 

RUN pip install scipy==1.3.3
RUN pip install requests==2.22.0
RUN pip install Pillow==6.2.1

# Install Addl Deps
RUN pip install IPython && \
    pip install imageio && \
    pip install keras==2.3.1 && \
    pip install dlib

RUN cd / && \
    git clone https://github.com/entmike/stylegan2 && \
    mkdir /models /in /out /latents /tmp /records && \
    cd /models && \
    wget http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl && \
    cd /stylegan2
