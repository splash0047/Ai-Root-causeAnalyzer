FROM python:3.10-slim

# Hugging Face Spaces run containers as a non-root user for security
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy requirements and install dependencies first (caches this step)
COPY --chown=user backend/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the backend codebase into the container
COPY --chown=user backend/ $HOME/app/

# Hugging Face Spaces expects the app to run on port 7860
EXPOSE 7860

# Command to run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
