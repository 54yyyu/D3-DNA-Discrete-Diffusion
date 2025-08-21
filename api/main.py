#!/usr/bin/env python3
"""
D3-DNA FastAPI Application

GPU-accelerated sequence generation API with simple web frontend
"""

import sys
import uvicorn
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.config import config
from api.models import (
    GenerationRequest, GenerationResponse, DatasetInfo, 
    ModelStatus, HealthResponse, ErrorResponse
)
from api.services.model_loader import model_loader
from api.services.generation import generation_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - preload models"""
    print("🚀 Starting D3-DNA API...")
    
    # Validate configuration
    validation_results = config.validate_paths()
    print(f"Configuration validation: {validation_results}")
    
    # Pre-load default models for faster first requests
    try:
        print("Pre-loading default DeepSTARR model...")
        dataset_config = config.get_dataset_config("deepstarr")
        model_loader.load_model("deepstarr", dataset_config)
        print(f"✓ Default model loaded successfully ({dataset_config['architecture']} architecture)")
    except Exception as e:
        print(f"⚠️ Failed to pre-load default model: {e}")
        print("Models will be loaded on first request")
    
    print(f"🌐 Server ready on http://{config.host}:{config.port}")
    
    yield
    
    # Cleanup
    print("🛑 Shutting down D3-DNA API...")
    model_loader.cleanup()


# Create FastAPI app
app = FastAPI(
    title="D3-DNA Sequence Generator",
    description="GPU-accelerated DNA sequence generation using discrete diffusion models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


# Exception handlers
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    print(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            status_code=500
        ).dict()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP error",
            detail=exc.detail,
            status_code=exc.status_code
        ).dict()
    )


# Main endpoints
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend page"""
    html_file = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_file.read_text(), status_code=200)


@app.post("/generate", response_model=GenerationResponse)
async def generate_sequences(request: GenerationRequest):
    """Generate DNA sequences with visualization data"""
    
    try:
        # Validate request
        if request.num_samples <= 0 or request.num_samples > config.max_samples_per_request:
            raise HTTPException(
                status_code=400, 
                detail=f"num_samples must be between 1 and {config.max_samples_per_request}"
            )
        
        if request.steps <= 0 or request.steps > config.max_steps_per_request:
            raise HTTPException(
                status_code=400,
                detail=f"steps must be between 1 and {config.max_steps_per_request}"
            )
        
        # Get dataset configuration
        try:
            dataset_config = config.get_dataset_config(request.dataset)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Generate sequences
        response = await generation_service.generate_sequences(request, dataset_config)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/datasets/info")
async def get_datasets_info():
    """Get information about available datasets"""
    
    try:
        datasets = config.get_available_datasets()
        
        dataset_list = []
        for name, info in datasets.items():
            dataset_list.append(DatasetInfo(
                name=name,
                sequence_length=info["sequence_length"],
                signal_dim=info["signal_dim"],
                num_classes=info["num_classes"],
                available_architectures=info["available_architectures"],
                default_steps=info["default_steps"],
                description=info["description"]
            ))
        
        return {"datasets": dataset_list}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset info: {str(e)}")


@app.get("/models/status")
async def get_model_status():
    """Get status of loaded models"""
    
    try:
        status = model_loader.get_model_status()
        
        model_statuses = []
        for model_info in status["models"]:
            model_statuses.append(ModelStatus(
                dataset=model_info["dataset"],
                architecture=model_info["architecture"],
                model_loaded=model_info["loaded"],
                oracle_loaded=model_info.get("oracle_loaded", False),
                device=status["device"],
                loading_time=model_info.get("loading_time"),
                error=model_info.get("error")
            ))
        
        return {
            "device": status["device"],
            "gpu_available": status["gpu_available"],
            "models": model_statuses,
            "memory_usage": status.get("gpu_memory")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    try:
        status = model_loader.get_model_status()
        
        model_statuses = []
        for model_info in status["models"]:
            model_statuses.append(ModelStatus(
                dataset=model_info["dataset"],
                architecture=model_info["architecture"],
                model_loaded=model_info["loaded"],
                oracle_loaded=model_info.get("oracle_loaded", False),
                device=status["device"],
                loading_time=model_info.get("loading_time"),
                error=model_info.get("error")
            ))
        
        return HealthResponse(
            status="healthy",
            gpu_available=status["gpu_available"],
            loaded_models=model_statuses,
            memory_usage=status.get("gpu_memory")
        )
        
    except Exception as e:
        print(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            gpu_available=False,
            loaded_models=[],
            memory_usage=None
        )


def main():
    """Main entry point"""
    print("🧬 D3-DNA Sequence Generator API")
    print(f"Device: {config.device}")
    print(f"Starting server on {config.host}:{config.port}")
    
    uvicorn.run(
        "api.main:app",
        host=config.host,
        port=config.port,
        reload=False,  # Disable reload for production
        access_log=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()