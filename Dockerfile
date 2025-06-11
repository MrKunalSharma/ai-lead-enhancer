# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy language model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application
COPY . .

# Create data directory
RUN mkdir -p /app/data

# Expose ports
EXPOSE 8501 8000

# Create a startup script
RUN echo '#!/bin/bash\n\
if [ "$1" = "api" ]; then\n\
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000\n\
else\n\
    streamlit run src/ui/streamlit_app.py\n\
fi' > /app/start.sh && chmod +x /app/start.sh

# Default command runs Streamlit
CMD ["/app/start.sh"]