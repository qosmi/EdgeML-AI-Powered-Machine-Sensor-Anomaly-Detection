"""FastAPI service for EdgeML."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipeline import analyze


app = FastAPI(
    title="EdgeML API",
    description="AI-powered machine sensor anomaly detection",
    version="1.0.0",
)


class AnalysisRequest(BaseModel):
    path: str


@app.get("/health")
def health() -> dict[str, str]:
    """Health check."""

    return {"status": "ok"}


@app.post("/analyze")
def analyze_sensor_data(
    request: AnalysisRequest,
) -> dict:
    """Analyze a sensor dataset."""

    try:
        result = analyze(request.path)

        return {
            "records": result["records"],
            "anomalies": result["anomalies"],
            "anomaly_rate": result["anomaly_rate"],
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc