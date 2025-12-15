import io
from fastapi import HTTPException, UploadFile
import pandas as pd


async def validate_file(file: UploadFile, required_columns: list) -> pd.DataFrame:
    contents = await file.read()
    if file.filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(contents))
    elif file.filename.endswith('.parquet'):    
        df = pd.read_parquet(io.BytesIO(contents))
    else:
        raise HTTPException(400, {"error": "Поддерживается загрузка только CSV или parquet файлов", "message": file.filename})
    if len(df) == 0:
        raise HTTPException(400, {"error": "Пустой файл", "message": file.filename})

    if not all(col in df.columns for col in required_columns):
        raise HTTPException(400, f"Обязательные поля: {required_columns}")
    
    return df