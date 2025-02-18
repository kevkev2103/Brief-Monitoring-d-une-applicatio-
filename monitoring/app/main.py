from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
import joblib
import pandas as pd
from pydantic import BaseModel
import os

# Obtenir les chemins absolus des fichiers
current_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_path = os.path.join(current_dir, "pipeline.joblib")
model_path = os.path.join(current_dir, "model.joblib")

# Charger le pipeline et le modèle
pipeline = joblib.load(pipeline_path)
model = joblib.load(model_path)


#initialisation de l'application FastAPI

app = FastAPI()

class InputData(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    island: str  
    sex: str

class OutputData(BaseModel):
    prediction: str    

class_mapping = {
    0: 'Adelie',
    1: 'Chinstrap',
    2: 'Gentoo'
}

#Endpoint pour prédire

@app.post("/predict/", response_model = OutputData)
async def predict(data:InputData):

    #convertir les données d'entrée en Dataframe
    input_data = pd.DataFrame([data.model_dump()])

    #etape 1: transformer les données avec le pipeline
    try:
        transformed_data = pipeline.transform(input_data)
    except Exception as e :
        raise HTTPException(status_code=400, detail=f"Erreur lors de la transformation des données : {e}")

    #étape 2 : Faire une prédiction avec le modèle
    try:
        prediction_encoded = model.predict(transformed_data)[0]
        

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {e}")
    
        # Étape 3 : Convertir la prédiction en texte à l'aide du dictionnaire
    prediction = class_mapping.get(prediction_encoded, "Classe inconnue")

    # Retourner la prédiction
    return {"prediction": prediction}


Instrumentator().instrument(app).expose(app)