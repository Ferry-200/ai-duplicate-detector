FROM python:3.12-slim

LABEL org.opencontainers.image.title="AI Duplicate Issue Detector"
LABEL org.opencontainers.image.description="Automatically detect duplicate and related GitHub issues using AI"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source files
COPY src/ ./src/

# Create cache directory
RUN mkdir -p .github/cache

# Set environment variables
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/app/src/detect_duplicates.py"] 