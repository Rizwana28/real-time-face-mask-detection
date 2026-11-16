FROM python:3.11-slim

# libglib is needed by OpenCV even in the headless build
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libxcb1 libgl1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HOST=0.0.0.0 \
    PORT=7860 \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics
WORKDIR /home/user/app

# CPU-only PyTorch keeps the image small and the build fast
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
EXPOSE 7860
CMD ["python", "app.py", "--no-browser"]
