from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np
import uvicorn

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = FastAPI(title="Customer Churn Prediction API")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(request: Request, features: str = Form(...)):
    try:
        features_list = [float(x.strip()) for x in features.split(",")]
        input_data = np.array(features_list).reshape(1, -1)
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        proba = model.predict_proba(scaled_data)[0][1]
        result_text = "Customer is likely to CHURN" if prediction == 1 else "Customer will NOT churn"
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": result_text,
            "probability": round(float(proba), 3)
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "result": f"Error: {str(e)}"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
