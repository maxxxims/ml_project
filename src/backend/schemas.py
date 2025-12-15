from pydantic import BaseModel, Field
from fastapi import UploadFile, File
from typing import Optional, List, Dict, Any
import base64
from pydantic import Field

class ForwardRequest(BaseModel):
    title: str
    description: str
    microcat_id: int
    item_price: Optional[float] = Field(default=-1, example=None)
    
class ForwardResponse(BaseModel):
    weight: float
    height: float
    length: float
    width: float


class ForwardBatchResponse(BaseModel):
    total: int
    item_ids: List[int]
    weights: List[float]
    heights: List[float]
    lengths: List[float]
    widths: List[float]
    
    
class EvalResponse(BaseModel):
    support: int
    metrics: Dict[str, float]
    

class ColdStorageResponse(BaseModel):
    rows: int
    columns: List[str]
    
class RetrainRequest(BaseModel):
    experiment_name: str = 'iwgh_prediction'
    max_train_iterations: int = 100
    
    
class ErrorResponse(BaseModel):
    error: str
    message: Optional[str] = None
    
    
class DeployRequest(BaseModel):
    run_id: str = ""