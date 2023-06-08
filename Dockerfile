FROM python:3.10.11
COPY . /exorde
WORKDIR /exorde
RUN apt-get update
RUN apt-get install zsh -y
RUN pip3.10 install /exorde/data
RUN pip3.10 install /exorde/data/scraping/reddit
RUN pip3.10 install /exorde/data/scraping/twitter
RUN pip3.10 install --upgrade git+https://github.com/JustAnotherArchivist/snscrape.git 
RUN pip3.10 install /exorde/lab
RUN pip3.10 install /exorde/exorde
RUN python3 -m spacy download en_core_web_trf
RUN install_translation
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python 
cp /exorde/exorde/exorde/protocol/base/configuration.yaml /usr/local/lib/python3.10/site-packages/exorde/protocol/base/
ENTRYPOINT ["/bin/zsh"]