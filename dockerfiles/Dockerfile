FROM apache/airflow:2.9.1-python3.9

# Switch to root user to run apt-get
USER root

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev \
    curl \
    unzip

# Clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update && sudo apt-get upgrade -y && sudo apt-get install -y git

# Switch back to airflow user
USER airflow

# Set work directory
WORKDIR /opt/airflow

# Install Python, pip and virtualenv
#RUN pip install --upgrade pip
RUN pip install virtualenv

# Copy requirements file and install dependencies
COPY requirements.txt /opt/airflow/
COPY setup.py /opt/airflow/

# Create and activate a virtual environment in /opt/airflow/venv
RUN virtualenv /opt/airflow/venv && \
    /opt/airflow/venv/bin/pip install --no-cache-dir -r /opt/airflow/requirements.txt 

RUN pip install scikit-surprise
RUN pip install mlflow==2.13.1

# Set environment variable for virtualenv
ENV PATH="/opt/airflow/venv/bin:$PATH"

# Activate virtual environment and install requirements
COPY requirements.txt /requirements.txt

CMD ["tail", "-f", "/dev/null"]