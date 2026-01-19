FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system build deps if needed
RUN apt-get update \
	&& apt-get install -y --no-install-recommends gcc \
	&& rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir pandas numpy scikit-learn xgboost fastapi uvicorn pydantic joblib

# Copy application files. If you trained locally, include model.pkl and preprocessor.pkl
COPY . .

# Expose port
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]