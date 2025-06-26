FROM python:3.9

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean

WORKDIR /app

COPY . /app
COPY data/06_models/model.pth data/06_models/model.pth
RUN test -f data/06_models/model.pth

COPY requirements-docker.txt requirements-docker.txt 
RUN ls -R /app
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements-docker.txt

CMD ["python", "src/asi_pro/gradio_app.py"]
