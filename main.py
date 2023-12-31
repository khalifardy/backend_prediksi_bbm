from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

class CarData(BaseModel):
    displacement: float
    horsepower: float
    weight: float
    acceleration: float
    cylinders: int
    origin: str
    model_year: int

@app.post("/process_car_data/")
async def process_car_data(car_data: CarData):
    # Lakukan pemrosesan atau logika bisnis di sini
    dictio_origin = {
        'USA':1,
        'EUROPE':2,
        'ASIA':3
    }
    dictio = car_data.model_dump()
    displacement_norm = (dictio['displacement']-68)/387
    horsepower_norm = (dictio['horsepower']-100)/130
    weight_norm = (dictio['weight']-1613)/3527
    acceleration_norm = (dictio['acceleration']-8)/16.8
    origin_diskrit = dictio_origin[dictio['origin']]

    list_hasil = [
        displacement_norm,
        horsepower_norm,
        weight_norm,
        acceleration_norm,
        dictio['cylinders'],
        origin_diskrit,
        dictio['model_year']
    ]
    

    arr = np.array(list_hasil)
    with open('boosting_regressi.pkl', 'rb') as file:
        mdl = pickle.load(file)
    
    hasil = mdl.predict(arr)

    result = {"hasil":hasil}
    return result
