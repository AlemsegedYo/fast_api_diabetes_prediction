# Use official Python image
FROM python:3.9

# Set work directory
WORKDIR /app

# Copy files
COPY ./app /app

# Install dependencies

COPY ./requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Create model directory
RUN mkdir -p /app/model

# Run training script
RUN python train_model.py

# Expose port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
