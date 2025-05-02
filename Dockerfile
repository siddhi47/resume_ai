# Use a Python base image
FROM python:3.9-slim

# Set environment variables to prevent Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
# Set environment variables to ensure that the output is displayed immediately in the console
ENV PYTHONUNBUFFERED 1

# Create and set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update \
  && apt-get install -y \
  build-essential \
  libpq-dev \
  && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file and install dependencies
COPY requirements.txt /app/

RUN pip install .

# Copy the entire project to the container
COPY . /app/

# Expose the Flask port (5000 by default)
EXPOSE 5000

# Set the environment variable for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Start the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]

