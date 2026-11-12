# Image de l'API : FastAPI seul, sans torch (l'inférence est faite par le conteneur vLLM)
# Construction depuis la racine du projet : docker build -f api/Dockerfile .
FROM python:3.12-slim

WORKDIR /app
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Le module partagé avec la préparation des données : même prompt qu'à l'entraînement
COPY src/projet_14/__init__.py src/projet_14/mediqal.py projet_14/
COPY api/main.py .

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
