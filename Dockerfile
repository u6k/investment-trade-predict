FROM python:3.7
LABEL maintainer="u6k.apps@gmail.com"

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get clean && \
    pip install pipenv

VOLUME /var/myapp
WORKDIR /var/myapp

CMD ["bash"]

