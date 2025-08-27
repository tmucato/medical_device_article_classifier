# Use a slim Python 3.12 image as a base to keep the image size small.
FROM python:3.12-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements file into the working directory.
# This step is placed here to leverage Docker's build cache.
# If only the application code changes, this layer won't be rebuilt.
COPY python/requirements.txt .

# Install the Python dependencies. The --no-cache-dir flag
# reduces the final image size by not storing the pip cache.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container (only the python app directory).
COPY python/ .

# Expose the port that the Python application will listen on.
# This should match the port you defined for the app service in docker-compose.yml.
EXPOSE 8000

# The command to run the application. Use the list format to ensure it runs correctly.
# This assumes your main application file is named app.py.
CMD ["python", "app.py"]
