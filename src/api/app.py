"""Production FastAPI microservice for real-time and batch house valuation."""
import time
import logging
import joblib
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from src.config import MODEL_ARTIFACT_PATH
from src.api.schemas import (
    HouseFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    FeatureExplanation
)
from src.models.explain import ModelExplainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# State storage for model and explainer
ml_state = {
    "pipeline": None,
    "explainer": None,
    "model_version": "1.0.0",
    "request_count": 0,
    "total_predictions": 0,
    "last_prediction_time_ms": 0.0
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads model pipeline and initializes TreeExplainer during server startup."""
    logger.info("Initializing ML serving application...")
    if MODEL_ARTIFACT_PATH.exists():
        try:
            pipeline = joblib.load(MODEL_ARTIFACT_PATH)
            ml_state["pipeline"] = pipeline
            logger.info("Model pipeline loaded successfully.")
            ml_state["explainer"] = ModelExplainer(pipeline)
        except Exception as e:
            logger.error(f"Failed to load model pipeline: {e}")
    else:
        logger.warning(f"Model artifact not found at {MODEL_ARTIFACT_PATH}. Serving in uninitialized mode.")
    yield
    logger.info("Shutting down ML serving application.")


app = FastAPI(
    title="Seattle King County House Price Prediction Service",
    description="Production-grade AI/ML microservice predicting residential valuations with explainability and demographics enrichment.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_model_loaded():
    if ml_state["pipeline"] is None and MODEL_ARTIFACT_PATH.exists():
        try:
            logger.info("Lazy-loading model pipeline from disk...")
            pipeline = joblib.load(MODEL_ARTIFACT_PATH)
            ml_state["pipeline"] = pipeline
            ml_state["explainer"] = ModelExplainer(pipeline)
        except Exception as e:
            logger.error(f"Error lazy-loading model: {e}")


def _predict_dataframe(df: pd.DataFrame, include_explanations: bool = False):
    _ensure_model_loaded()
    pipeline = ml_state["pipeline"]
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Train the model pipeline first."
        )

    t0 = time.time()
    preds = pipeline.predict(df)
    elapsed_ms = (time.time() - t0) * 1000.0

    ml_state["request_count"] += 1
    ml_state["total_predictions"] += len(df)
    ml_state["last_prediction_time_ms"] = elapsed_ms

    results = []
    # Assumes typical residual error margin of ~12% (based on test MAPE) for 90% confidence bounds
    error_margin_ratio = 0.12

    explainer = ml_state["explainer"]

    for i, pred_val in enumerate(preds):
        price = float(np.round(pred_val, 2))
        lower_bound = float(np.round(price * (1.0 - error_margin_ratio), 2))
        upper_bound = float(np.round(price * (1.0 + error_margin_ratio), 2))

        explanations = None
        if include_explanations and explainer is not None:
            single_row = df.iloc[[i]]
            raw_expl = explainer.explain_instance(single_row, top_k=5)
            explanations = [FeatureExplanation(**item) for item in raw_expl]

        results.append(
            PredictionResponse(
                predicted_price=price,
                predicted_price_formatted=f"${price:,.2f}",
                confidence_interval_90_usd={
                    "lower_bound_usd": lower_bound,
                    "upper_bound_usd": upper_bound
                },
                model_version=ml_state["model_version"],
                explanations=explanations
            )
        )
    return results


@app.get("/", tags=["General"])
async def root():
    return {
        "service": "Seattle House Price Prediction API",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
        "metrics_url": "/metrics"
    }


@app.get("/health", response_model=HealthResponse, tags=["Observability"])
async def health():
    _ensure_model_loaded()
    return HealthResponse(
        status="healthy" if ml_state["pipeline"] is not None else "degraded",
        model_version=ml_state["model_version"],
        model_loaded=ml_state["pipeline"] is not None,
        pipeline_type="Scikit-Learn TransformedTargetRegressor + LightGBM"
    )


@app.get("/metrics", response_class=PlainTextResponse, tags=["Observability"])
async def metrics():
    """Prometheus-compatible plain text metrics."""
    return f"""# HELP app_requests_total Total number of inference requests
# TYPE app_requests_total counter
app_requests_total {ml_state["request_count"]}

# HELP app_predictions_total Total number of house predictions rendered
# TYPE app_predictions_total counter
app_predictions_total {ml_state["total_predictions"]}

# HELP app_last_prediction_latency_ms Latency of the most recent prediction in milliseconds
# TYPE app_last_prediction_latency_ms gauge
app_last_prediction_latency_ms {ml_state["last_prediction_time_ms"]:.2f}
"""


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_single(house: HouseFeatures):
    """Predict market value for a single residential property."""
    df_input = pd.DataFrame([house.model_dump()])
    res = _predict_dataframe(df_input, include_explanations=False)
    return res[0]


@app.post("/predict/explain", response_model=PredictionResponse, tags=["Inference"])
async def predict_with_explanation(house: HouseFeatures):
    """Predict market value along with top 5 feature SHAP attribution drivers."""
    df_input = pd.DataFrame([house.model_dump()])
    res = _predict_dataframe(df_input, include_explanations=True)
    return res[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(request: BatchPredictionRequest):
    """Predict market values for a batch of residential properties."""
    if not request.houses:
        raise HTTPException(status_code=400, detail="Empty houses list provided.")
    df_input = pd.DataFrame([h.model_dump() for h in request.houses])
    results = _predict_dataframe(df_input, include_explanations=False)
    return BatchPredictionResponse(
        total_properties=len(results),
        predictions=results
    )
