from fastapi import APIRouter, Depends
import numpy as np

from customer_churn.api.deps import get_predictor
from customer_churn.ml.inference.predictor import ChurnPredictor
from customer_churn.api.ml.schemas import CustomerData, PredictionResponse
from customer_churn.ml.data.utils import prepare_features


router = APIRouter(prefix="/model", tags=["MODEL"])


@router.post("/predict", response_model=PredictionResponse)
async def get_prediction(
    customer: CustomerData, predictor: ChurnPredictor = Depends(get_predictor)
) -> PredictionResponse:
    """
    Predict customer churn probability based on customer data.

    Args:
        customer: Customer data input
        predictor: ChurnPredictor instance

    Returns:
        Prediction response with churn probability and binary prediction
    """
    features = prepare_features(customer).reshape(1, -1)
    churn_prob = float(predictor.predict(features))

    return PredictionResponse(
        churn_probability=churn_prob, is_likely_to_churn=churn_prob > 0.5
    )
