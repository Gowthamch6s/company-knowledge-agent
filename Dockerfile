FROM python:3.13-slim

WORKDIR /app

# Install required system packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first so Docker can cache this layer
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app ./app
COPY static ./static
COPY documents ./documents

# FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]