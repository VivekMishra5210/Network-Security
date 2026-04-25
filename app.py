import sys
import os
import certifi
import pandas as pd
import pymongo

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from uvicorn import run as app_run

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from networksecurity.constant.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME
)

# ------------------ ENV + DB SETUP ------------------
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL_KEY")
print("Mongo URL:", mongo_db_url)

ca = certifi.where()

client = None
database = None
collection = None

if mongo_db_url:
    try:
        client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
        database = client[DATA_INGESTION_DATABASE_NAME]
        collection = database[DATA_INGESTION_COLLECTION_NAME]
        print("✅ MongoDB connected")
    except Exception as e:
        print("❌ MongoDB connection failed:", e)
else:
    print("⚠️ MongoDB URL not set — skipping DB connection")

# ------------------ FASTAPI SETUP ------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ ROUTES ------------------

@app.get("/")
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise NetworkSecurityException(e, sys)


@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    try:
        # Step 1: Read input
        df = pd.read_csv(file.file)

        # Step 2: Load model
        preprocessor = load_object("final_model/preprocessor.pkl")
        model = load_object("final_model/model.pkl")

        # Step 3: Predict
        network_model = NetworkModel(preprocessor=preprocessor, model=model)
        y_pred = network_model.predict(df)

        # Step 4: Add predictions
        df["predicted_column"] = y_pred

        # Step 5: Save output
        os.makedirs("prediction_output", exist_ok=True)
        df.to_csv("prediction_output/output.csv", index=False)

        # Step 6: Convert to HTML
        table_html = df.to_html(classes="table table-striped")

        # Step 7: Return HTML directly (NO JINJA)
        return HTMLResponse(content=f"""
        <html>
            <head>
                <title>Prediction Results</title>
                <style>
                    table {{
                        border-collapse: collapse;
                        width: 100%;
                    }}
                    th, td {{
                        border: 1px solid black;
                        padding: 8px;
                        text-align: left;
                    }}
                </style>
            </head>
            <body>
                <h2>Predicted Data</h2>
                {table_html}
            </body>
        </html>
        """)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise NetworkSecurityException(e, sys)


# ------------------ MAIN ------------------
if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)