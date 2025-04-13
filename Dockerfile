FROM python:3.12.4-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies required for building grpcio & protobuf
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libprotobuf-dev \
    libgrpc++-dev \
    pkg-config \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Upgrade pip tooling to ensure wheel builds succeed
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose port for Streamlit
EXPOSE 8080

# Run the application
CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
