FROM cytomineuliege/software-python3-base:latest

RUN /bin/sh -c set -ex && wget -O neubiaswg5-util.tar.gz https://github.com/Neubias-WG5/neubiaswg5-utilities/archive/v0.8.0.tar.gz && tar xf neubiaswg5-util.tar.gz --strip 1 && pip install .

RUN pip install -r requirements.txt

RUN mkdir -p /app

ADD run.py /app/run.py

ADD segmentation /app/segmentation

ENTRYPOINT ["python", "/app/run.py"]