FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p checkpoints results visualizations logs data_cache

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Make run script executable
RUN chmod +x run.sh || echo "run.sh not found, skipping"

# Default command - trains all models
CMD ["python", "train.py", "--config", "config_quick.yaml", "--model", "all"]
