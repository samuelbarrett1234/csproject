FROM continuumio/miniconda3
USER root
# NOTE: prevent interaction with tzdata install:
ARG DEBIAN_FRONTEND=noninteractive


RUN echo "APT INSTALLS" \
	&& apt-get update \
    && apt-get -qq -y install git curl unzip sqlite3 wget

# see https://pythonspeed.com/articles/activate-conda-dockerfile/
RUN echo "CONDA ENV SETUP" \
    && conda create --name ori_ade python=3.7 \
    && conda init bash \
    && echo "conda activate ori_ade" > ~/.bashrc


RUN echo "PYTHON INSTALLS" \
    && conda install -y tensorflow-gpu==2.3.0 progressbar2 numpy pytest nltk cython \
    && conda install -y -c huggingface transformers


RUN echo "REPOSITORY SETUP" \
    && git clone https://github.com/samuelbarrett1234/csproject
