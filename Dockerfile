FROM python:3.12-slim

# pyspark runs on the jvm so we need a jdk. headless keeps the image smaller.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jdk-headless procps bash && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /bin/bash /bin/sh

# default-java is an arch independent symlink, so this works on amd64 and arm64
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# default run is the full medallion backfill plus the train / val / test split.
# remember to run generate_data.py on the host first so data/ exists.
CMD ["python", "main.py"]
