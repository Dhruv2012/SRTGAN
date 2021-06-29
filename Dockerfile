FROM continuumio/anaconda3
RUN apt-get update \
    && apt-get install -y libgtk2.0-dev
WORKDIR /app
RUN apt-get install -y git
RUN git clone https://github.com/Dhruv2012/Image-SuperResolution.git
RUN git checkout supervised
RUN cd env_yamls
RUN conda env create -f superres_cuda11.yml
