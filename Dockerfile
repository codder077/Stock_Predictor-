FROM python:3.14.0-slim

# Prevents Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Ensures logs are streamed to the console immediately
ENV PYTHONUNBUFFERED=1

# Set Working Directory
WORKDIR /app

# Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data
COPY . .

# Train the Model
RUN python -m src.train

# Expose API Port
EXPOSE 8000

# Start the FastAPI Server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]