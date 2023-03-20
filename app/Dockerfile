FROM ubuntu:22.04

EXPOSE 8001

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y

RUN apt update -y && apt install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx

RUN apt-get install libglib2.0-0 -y

RUN apt install python3-pip -y


# RUN cd Python-3.10.*/ && \
#      ./configure --enable-optimizations && \
#      make -j $(nproc) && \
#      make install


# RUN curl -Lo /usr/local/bin/bazel https://github.com/bazelbuild/bazel/releases/download/5.2.0/bazel-5.2.0-linux-arm64

# RUN chmod +x /usr/local/bin/bazel


# RUN git clone https://github.com/google/mediapipe.git


# COPY library/mediapipe ./mediapipe

# RUN cd ./mediapipe && \
#     pip install -r requirements.txt && \

RUN apt-get install ffmpeg -y

RUN apt-get clean && \ 
    rm -rf /var/lib/apt/lists/*


# WORKDIR /app

COPY library/mediapipe-0.9.1-cp310-cp310-linux_aarch64.whl ./

CMD ls library/

CMD python3 --version

RUN pip install flask flask_cors tensorflow numpy opencv-python-headless


RUN pip install ./mediapipe-0.9.1-cp310-cp310-linux_aarch64.whl

COPY ./app/ /app/

COPY ./models/ /models/

CMD ["python3", "app/app.py"]
