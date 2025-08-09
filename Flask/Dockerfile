# Use official Python 3.10 image
FROM python:3.10

# Install ffmpeg, PortAudio (for sounddevice), and other system packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    portaudio19-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd -m -u 1000 user
USER user

# Add user’s local bin to PATH for pip-installed executables
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory inside container
WORKDIR /app

# Copy and install Python dependencies as non-root user
COPY --chown=user ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application code
COPY --chown=user . /app

# Expose the port your app will run on
EXPOSE 7860

# Start Gunicorn server binding to 0.0.0.0 on port 7860
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:7860"]
