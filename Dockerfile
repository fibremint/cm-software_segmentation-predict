FROM cytomineuliege/software-python3-base:latest

RUN /bin/sh -c set -ex && wget -O /tmp/neubiaswg5-util.tar.gz https://github.com/Neubias-WG5/neubiaswg5-utilities/archive/v0.8.0.tar.gz && tar -xf /tmp/neubiaswg5-util.tar.gz --strip 1 -C /tmp && pip install /tmp/.

COPY requirements.txt /tmp

RUN pip install -r /tmp/requirements.txt

RUN mkdir -p /app

ADD run.py /app/run.py

ADD segmentation /app/segmentation

ENTRYPOINT ["python", "/app/run.py"]
