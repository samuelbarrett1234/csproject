FROM tensorflow/tensorflow:2.3.0-gpu

ENV HF_HOME="/data/csproject/hf_cache_home"

RUN echo "APT INSTALLS" \
	&& apt-get update --allow-releaseinfo-change \
    && apt-get -qq -y install git curl unzip sqlite3 wget

RUN echo "PYTHON INSTALLS" \
    && pip install progressbar2 pytest nltk cython transformers==4.10.2 \
    spacy miditok miditoolkit matplotlib scikit-learn networkx seaborn \
    numpy==1.19.2 scipy jax==0.2.8 dm-haiku==0.0.4 more_itertools==8.3.0 jaxlib

RUN echo "NLTK setup" \
    && python -m nltk.downloader punkt reuters \
    && cp -R /root/nltk_data /usr/local/share/nltk_data

#RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test \
#    && apt-get -qq -y install cmake make build-essential libsqlite3-dev gcc g++-11 \
#    && ln -s /usr/bin/make /usr/bin/gmake
