from fastapi import FastAPI, UploadFile, File

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    return {
        "label": "Cat",
        "confidence": 0.95,
        "probabilities": {
            "Cat": 0.95,
            "Dog": 0.05
        }
    }
