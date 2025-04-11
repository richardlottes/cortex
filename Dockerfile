FROM python:3.12.4

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install Google Cloud Secret Manager client
RUN pip install --no-cache-dir google-cloud-secret-manager

# Copy project
COPY . .

# Expose port
EXPOSE 8080

# Run the application
CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.enableCORS=false"]
