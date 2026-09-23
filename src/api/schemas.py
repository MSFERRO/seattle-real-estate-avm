"""Pydantic V2 schemas for API contract validation."""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import date


class HouseFeatures(BaseModel):
    """Input payload validating physical and geographic house features."""
    bedrooms: int = Field(..., ge=0, le=33, description="Número de quartos")
    bathrooms: float = Field(..., ge=0.0, le=15.0, description="Número de banheiros")
    sqft_living: int = Field(..., gt=0, le=25000, description="Área habitável interna em pés²")
    sqft_lot: int = Field(..., gt=0, description="Área total do terreno em pés²")
    floors: float = Field(..., ge=1.0, le=5.0, description="Número de andares")
    waterfront: int = Field(0, ge=0, le=1, description="Vista para água (0=Não, 1=Sim)")
    view: int = Field(0, ge=0, le=4, description="Qualidade da vista de 0 a 4")
    condition: int = Field(3, ge=1, le=5, description="Estado de conservação de 1 a 5")
    grade: int = Field(7, ge=1, le=13, description="Padrão construtivo e acabamento de 1 a 13")
    sqft_above: int = Field(..., ge=0, description="Área construída acima do solo em pés²")
    sqft_basement: int = Field(0, ge=0, description="Área construída de porão em pés²")
    yr_built: int = Field(..., ge=1800, le=2026, description="Ano de construção")
    yr_renovated: int = Field(0, ge=0, le=2026, description="Ano de reforma (0 se nunca reformado)")
    zipcode: int = Field(..., description="Código postal residencial do imóvel")
    lat: float = Field(..., ge=46.5, le=48.5, description="Latitude geográfica (King County)")
    long: float = Field(..., ge=-123.5, le=-120.5, description="Longitude geográfica (King County)")
    sqft_living15: int = Field(..., gt=0, description="Área habitável média dos 15 vizinhos mais próximos")
    sqft_lot15: int = Field(..., gt=0, description="Área de terreno média dos 15 vizinhos mais próximos")
    valuation_date: Optional[str] = Field(None, description="Data de avaliação (YYYY-MM-DD), default=ano de referência")

    @field_validator("bedrooms")
    @classmethod
    def correct_known_bedroom_typo(cls, v: int) -> int:
        # Gracefully handle the 33 bedrooms typo in API requests
        return 3 if v == 33 else v


class FeatureExplanation(BaseModel):
    feature: str
    value: Any
    impact_percentage: float
    direction: str
    summary: str


class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Preço estimado do imóvel em dólares (USD)")
    predicted_price_formatted: str = Field(..., description="Preço estimado formatado")
    confidence_interval_90_usd: Dict[str, float] = Field(..., description="Intervalo de confiança de 90% (USD)")
    model_version: str
    explanations: Optional[List[FeatureExplanation]] = None


class BatchPredictionRequest(BaseModel):
    houses: List[HouseFeatures]


class BatchPredictionResponse(BaseModel):
    total_properties: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool
    pipeline_type: str
