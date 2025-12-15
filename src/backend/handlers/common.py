import io
from typing import Optional
from fastapi import APIRouter, Body, File, HTTPException, Depends, UploadFile
import time
import pandas as pd 
import logging

from src.backend.schemas import ErrorResponse
from src.core.models import Predictor
from src.backend.dependencies import (
    get_predictor,
)

router = APIRouter(tags=["common"])


@router.get(
    "/metadata",
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def get_metadata(predictor: Predictor = Depends(get_predictor)):
    try:
        return predictor.metadata
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal server error", "message": str(e)}
        )
        
        
@router.get(
    "/microcat_name",
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def get_microcat_info(microcat_id: int, predictor: Predictor = Depends(get_predictor)):
    """
    По microcat_id возвращает название (путь в дереве микрокатов) и среднюю цену в микрокате
    """
    microcat_id = int(microcat_id)
    if microcat_id not in predictor.microcats:
        raise HTTPException(
            status_code=400,
            detail={"error": "Unknown microcat_id", "microcat_id": microcat_id}
        )
    microcat_path = predictor._get_microcat_name(microcat_id)
    microcat_mean_price = predictor.microcat_2_price[microcat_id]
    
    return {'microcat_name': microcat_path, 'microcat_mean_price': microcat_mean_price}
