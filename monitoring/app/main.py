from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel

#chargement du modele
pipeline = joblib.load("pipeline.pkl")
model = joblib.load("model.pkl")
target_encoder = joblib.load("target_encoder.pkl")

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
    input_data = pd.DataFrame([data.dict()])

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


