FROM python:3.9-slim

# Set environment variable for pip timeout
ENV PIP_DEFAULT_TIMEOUT=100

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the application
CMD ["python", "satsifaction_score_predictor.py"]
