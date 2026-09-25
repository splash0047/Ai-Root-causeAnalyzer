FROM python:3.10-slim

RUN useradd -m -u 1000 appuser
WORKDIR /home/appuser/app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY --chown=appuser backend/ backend/
COPY --chown=appuser model/ model/
RUN mkdir -p backend/data && chown -R appuser:appuser backend model
USER appuser
# Generate the deterministic synthetic baseline in the image.
RUN python model/train_baseline.py
ENV PYTHONPATH=/home/appuser/app/backend \
    DATABASE_URL=sqlite:////home/appuser/app/backend/rca_local.db
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
