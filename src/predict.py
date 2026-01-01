from fastapi import FastAPI
import uvicorn

app = FastAPI()
@app.post("/predict")
def predict():
    return {"slice_id": 1}

if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=8080)
