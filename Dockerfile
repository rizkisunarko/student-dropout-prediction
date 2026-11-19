FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment
ENV PYTHONPATH=.
ENV PORT=5000

EXPOSE 5000

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app.main:app
