# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install torch==1.13.1


# Expose port 80 for the Flask web server
EXPOSE 8000

# Set the entrypoint to the Python script that runs the Flask web server
CMD ["streamlit", "run", "main.py", "--server.port=8000", "--server.address=localhost"]
