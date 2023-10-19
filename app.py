from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the trained logistic regression model
logistic_regression_model = joblib.load('logistic_regression_model.pkl')

class CustomerData(BaseModel):
    Age: int
    Subscription_Length_Months: int
    Monthly_Bill: float
    Total_Usage_GB: float
    Gender_Female: int
    Gender_Male: int
    Location_Chicago: int
    Location_Houston: int
    Location_Los_Angeles: int
    Location_Miami: int
    Location_New_York: int

@app.post("/predict_churn/")
async def predict_churn(data: CustomerData):
    input_data = data.dict()
    input_features = np.array(list(input_data.values())).reshape(1, -1)
    prediction = logistic_regression_model.predict(input_features)
    return {"churn_prediction": int(prediction[0])}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)