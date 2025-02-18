# Utilise une image Python légère
FROM python:3.9-slim

# Définit le répertoire de travail dans le conteneur
WORKDIR /app

# Copie les fichiers nécessaires
COPY . /app/
COPY pipeline.joblib /app/pipeline.joblib

# Installe les dépendances Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose le port utilisé par l'API
EXPOSE 8000

# Commande pour lancer l'API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
