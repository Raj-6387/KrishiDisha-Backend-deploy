
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from typing import List, Union
from fastapi.middleware.cors import CORSMiddleware
from models import get_crop_model,get_fertilizer_model,get_input


description = """
### Crop Recommendation JSON Input 
    { "array": [N,P,K,temperature,humidity,ph,rainfall] }
### Fertilizer Recommendation JSON Input 
    { "array": [Temparature,Humidity,Moisture,Nitrogen,Potassium,Phosphorous,Soil Type,Crop Type] }
"""

app = FastAPI(description=description)

# ------------------------------------------

# Enabling CORS policy

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained models
crop_model = get_crop_model()
fertilizer_model = get_fertilizer_model()

#--------------------------------------------------------------------

class InputArray(BaseModel):
    array: List[Union[float, str]] = Field(..., description="An array of floats or strings")

@app.post("/crop_recommend")
def crop_recommend_endpoint(request: Request, input: InputArray):
    
    print("Making Crop Prediction...")
    # Get probabilities
    probabilities = crop_model.predict_proba([input.array])[0]
    
    # Get the classes
    classes = crop_model.classes_
    
    # Pair classes with probabilities
    prob_class_pairs = list(zip(classes, probabilities))
    
    # Sort in descending order of probability
    prob_class_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Extract top 4 crops
    top_crops = [pair[0] for pair in prob_class_pairs[:4]] 
    
    predicted_crop = top_crops[0]
    optional_crops = top_crops[1:4]
    
    print("Returning Response...")    
    # Return the prediction in the response
    return {
        "predicted_crop": predicted_crop,
        "optional_crops": optional_crops
    }


@app.post("/fertilizer_recommend")
def array_endpoint(request: Request, input: InputArray):
    
    # process the input array
    x = get_input(input.array)
    
    print("Making Fertilizer Prediction...")
    # Make a prediction using the input array
    prediction = fertilizer_model.predict([x])
    prediction = prediction[0]
    
    print("Returning Response...")    
    # Return the prediction in the response
    return prediction

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)
    
 
