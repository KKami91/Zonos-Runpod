FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Install system dependencies
RUN apt-get update && \
    apt-get install -y espeak-ng git && \
    rm -rf /var/lib/apt/lists/*

# Install pip and uv
RUN pip install --upgrade pip && \
    pip install uv runpod

# Set working directory
WORKDIR /app

# Clone Zonos repository
RUN git clone https://github.com/KKami91/Zonos-Runpod.git /app && \
    cd /app

# Install Zonos and its dependencies
RUN uv pip install --system -e . && \
    uv pip install --system -e .[compile]

# Copy the handler file
COPY handler.py /app/handler.py

# Expose the port that RunPod uses
EXPOSE 8000

# Start the RunPod handler
CMD ["python", "handler.py"]