"""FastAPI Application Main Entrypoint.

Initializes FastAPI app instance, configures CORS middleware, lifespan setup,
registers API route handlers, and sets up exception handlers.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router as api_router
from app.config import setup_logging, settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager handling startup and shutdown events."""
    setup_logging(settings.LOG_LEVEL)
    logger.info("=========================================================================")
    logger.info("Starting Production Machine Learning RAG Agent Service (%s)", settings.PROJECT_NAME)
    logger.info("Environment: %s | Host: %s:%d", settings.ENVIRONMENT, settings.HOST, settings.PORT)
    logger.info("=========================================================================")
    yield
    logger.info("Shutting down Machine Learning RAG Agent Service...")


def create_app() -> FastAPI:
    """Factory function creating and configuring the FastAPI application instance.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="Production-grade Retrieval-Augmented Generation (RAG) Agent API for Machine Learning documents.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Configure CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routes
    app.include_router(api_router)

    # Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled global error on request %s: %s", request.url.path, exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred processing your request.",
                "details": str(exc) if settings.ENVIRONMENT == "development" else None,
            },
        )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True if settings.ENVIRONMENT == "development" else False,
    )
