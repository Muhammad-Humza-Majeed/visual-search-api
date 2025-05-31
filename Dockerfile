# Use an official Python runtime as a parent image
# We use a specific version to ensure consistency
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for Pillow, etc.
# These are often required for image processing libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgpl-dev \
    # Clean up APT when done
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We use --no-cache-dir to save space
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# This copies app.py, similarity_module.py, .env, faiss_index.bin, cid_map.npy, etc.
# Everything in your project root, except what's specified in .dockerignore
COPY . .

# IMPORTANT: Set the KMP_DUPLICATE_LIB_OK environment variable
# This resolves the OMP: Error #15 issue you encountered locally
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Expose the port your Flask app will run on. Cloud Run expects port 8080 by default.
# The `PORT` environment variable will be set by Cloud Run.
ENV PORT 8080

# Run the application
# Use Gunicorn for production-ready deployment
# Gunicorn is a WSGI HTTP Server for UNIX. Flask's built-in server is for development only.
# Install gunicorn in your requirements.txt if you haven't already: pip install gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app