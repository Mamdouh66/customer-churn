from customer_churn.api.core.model import ModelManager


def get_predictor():
    """Dependency for getting model predictor instance."""
    return ModelManager.get_predictor()
