import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import datetime
from prometheus_client import make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
import logging

from src.backend.handlers import forward, pipline, common
from src.backend.dependencies import engine

logger = logging.getLogger(__name__)
    

app = FastAPI(
    title="Avito Items Weight & Dimensions Prediction API",
    description="API для предсказания веса и габаритов товаров почти в Авито",
    version="1.0.0",
)

metrics_app = make_asgi_app()


instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    env_var_name="ENABLE_METRICS",
    excluded_handlers=["/metrics", "/health", "/"],
    inprogress_name="ml_api_inprogress",
    inprogress_labels=True,
)


instrumentator.instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(common.router)
app.include_router(forward.router)
app.include_router(pipline.router)


@app.get("/health", tags=['system_internal'])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

@app.get("/metrics", tags=['system_internal'])
async def metrics():
    """Endpoint for Prometheus"""
    return metrics_app


# @app.middleware("http")
# async def add_custom_metrics(request: Request, call_next):
#     start_time = time.time()
    
#     response = await call_next(request)
    
#     # Логируем запросы к API
#     if request.url.path.startswith("/forward"):
#         process_time = time.time() - start_time
        
#         # Эти метрики будут автоматически собраны instrumentator
#         # после того как response вернется
        
#     return response


def healchecker():
    import urllib.request
    urllib.request.urlopen('http://localhost:8000/health')
    
if __name__ == "__main__":
    healchecker()