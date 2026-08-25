# Use a Python base image
FROM python:3.12-slim

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
  curl \
  ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install tectonic (self-contained LaTeX engine) for compiling tailored resumes to PDF
RUN TECTONIC_VERSION=0.17.0 \
  && case "$(dpkg --print-architecture)" in \
       arm64) TECTONIC_ARCH=aarch64-unknown-linux-musl ;; \
       amd64) TECTONIC_ARCH=x86_64-unknown-linux-musl ;; \
       *) echo "Unsupported architecture: $(dpkg --print-architecture)" && exit 1 ;; \
     esac \
  && curl -fsSL -o /tmp/tectonic.tar.gz \
       "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40${TECTONIC_VERSION}/tectonic-${TECTONIC_VERSION}-${TECTONIC_ARCH}.tar.gz" \
  && tar -xzf /tmp/tectonic.tar.gz -C /usr/local/bin \
  && chmod +x /usr/local/bin/tectonic \
  && rm /tmp/tectonic.tar.gz

# Copy the requirements.txt file and install dependencies

COPY . /app/
RUN pip install .

# Copy the entire project to the container

# Expose the Flask port (5000 by default)
EXPOSE 5050

# Set the environment variable for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Start the Flask app

CMD ["gunicorn", "-b", "0.0.0.0:5050", "--timeout", "480", "app:app"]
