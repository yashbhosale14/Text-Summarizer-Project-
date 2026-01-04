from fastapi import FastAPI
import uvicorn
import os
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from textSummarizer.pipeline.prediction import PredictionPipeline


app = FastAPI(title="Text Summarization API")

# ✅ Load model once
pipeline = PredictionPipeline()


@app.get("/", tags=["general"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train", tags=["training"])
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/predict", tags=["prediction"])
async def predict(text: str):
    # Clean input text
    clean_text = text.replace("\n", " ").replace("\r", " ")

    # ✅ Correct usage
    summary = pipeline.predict(clean_text)

    # 🔥 Final output cleaning
    summary = (
        summary
        .replace("<n>", " ")
        .replace("  ", " ")
        .strip()
    )

    return {"summary": summary}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
