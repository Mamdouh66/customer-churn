from contextlib import asynccontextmanager

from customer_churn.logger import get_logger
from customer_churn.api.deps import ModelManager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    logger.info("Loading model...")
    try:
        ModelManager.get_predictor()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

    yield

    logger.info("Shutting down application...")
    ModelManager._instance = None


def get_application() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    return app


app = get_application()
