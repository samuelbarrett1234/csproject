FROM tensorflow/tensorflow:2.3.0-gpu
USER root
# NOTE: prevent interaction with tzdata install:
ARG DEBIAN_FRONTEND=noninteractive


RUN echo "APT INSTALLS" \
	&& apt-get update --allow-releaseinfo-change \
    && apt-get -qq -y install git curl unzip sqlite3 wget

RUN echo "PYTHON INSTALLS" \
    && pip install progressbar2 numpy pytest nltk cython transformers


RUN echo "REPOSITORY SETUP" \
    && git clone https://github.com/samuelbarrett1234/csproject \
    && cd csproject \
    && python compressors/setup.py build_ext --inplace
