# Reproducible, OS-independent way to run the dashboard.
# Build:  docker build -t churn-app .
# Run:    docker run -p 8501:8501 churn-app
# Then open http://localhost:8501 in your browser.

FROM python:3.11-slim

# System libs needed by matplotlib / shap / numpy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Bind to all interfaces so the port is reachable from the host
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
