FROM cytomineuliege/software-python3-base:latest

RUN /bin/sh -c set -ex && wget -O /tmp/neubiaswg5-util.tar.gz https://github.com/Neubias-WG5/neubiaswg5-utilities/archive/v0.8.6.tar.gz && tar -xf /tmp/neubiaswg5-util.tar.gz --strip 1 -C /tmp && pip install /tmp/.
RUN apt-get update && apt-get install -y --no-install-recommends openslide-tools && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

RUN mkdir -p /app
ADD model /app/model
ADD segmentation /app/segmentation
ADD descriptor.json /app/descriptor.json
ADD run.py /app/run.py

ENTRYPOINT ["python", "/app/run.py"]
