FROM tensorflow/tensorflow:2.3.0-gpu

ENV HF_HOME="/data/csproject/hf_cache_home"

RUN echo "APT INSTALLS" \
	&& apt-get update --allow-releaseinfo-change \
    && apt-get -qq -y install git curl unzip sqlite3 wget

RUN echo "PYTHON INSTALLS" \
    && pip install progressbar2 numpy pytest nltk cython transformers
