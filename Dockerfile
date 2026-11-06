# Runs the Gradio demo on CPU. The model downloads from the Hub on the first request.
FROM python:3.11-slim

# The prebuilt llama.cpp library needs libgomp, which the slim image leaves out.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY spaces/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gradio==6.31.0

COPY config.py .
COPY app/ app/

# Listen on all interfaces, so the demo is reachable from outside the container.
ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 7860
CMD ["python", "-m", "app.space"]
