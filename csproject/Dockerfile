FROM tensorflow/tensorflow:2.3.0-gpu

ENV HF_HOME="/data/csproject/hf_cache_home"

RUN echo "APT INSTALLS" \
	&& apt-get update --allow-releaseinfo-change \
    && apt-get -qq -y install git curl unzip sqlite3 wget gcc g++

RUN echo "PYTHON INSTALLS" \
    && pip install progressbar2 numpy pytest nltk cython transformers==4.10.2 scipy spacy miditok miditoolkit matplotlib scikit-learn networkx seaborn

RUN echo "NLTK setup" \
    && python -m nltk.downloader punkt reuters \
    && cp -R /root/nltk_data /usr/local/share/nltk_data
