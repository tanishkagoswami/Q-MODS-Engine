# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the backend requirements first (for caching)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code into the container
COPY backend/ ./backend/

# Move into the backend directory so Uvicorn can find api.main
WORKDIR /app/backend

# Expose the FastAPI port
EXPOSE 8000

# Command to boot the Uvicorn server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]