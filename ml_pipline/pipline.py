import os
from pathlib import Path
from catboost import CatBoostRegressor
from matplotlib import pyplot as plt
import mlflow
import logging
import sys
import argparse

import pandas as pd

from src.core.models import Predictor
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
from typing import Dict, Any
import json
from src import extract_data, train_models, inference_models
from src.core.config import mlflow_client, MLFlowConfig


logger = logging.getLogger(__name__)


class MLPipeline:
    """Основной класс pipeline для обучение, инференса моделей и загрузки в MLFLOW"""
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name if experiment_name is not None else MLFlowConfig.experiment_name
        self.client = mlflow_client

    
    def __prepare_extracted_data(self, metadata: dict, df: pd.DataFrame, name: str, save_folder: Path):
        metadata[f"{name}_samples"] = len(df)
        if name == 'train':
            sample_data = df.head(500)
            sample_path = str(save_folder / "train_sample.csv")
            sample_data.to_csv(sample_path)
            mlflow.log_artifact(sample_path, "extract_data")
        output_data_path = str(save_folder / f"{name}_df_features.parquet")
        mlflow.log_param(f"{name}_output_data_path", output_data_path)
        df.to_parquet(output_data_path)
        
        for col in ['real_weight', 'real_height', 'real_length', 'real_width']:
            plt.figure(figsize=(10, 6))
            df[col].hist(bins=50)
            plt.title(f"{col} distribution")
            plt.xlim([0, df[col].quantile(0.98)])
            plt.xlabel(col)
            plt.ylabel("Frequency")
            dist_plot_path = str(save_folder / f"{col}_distribution.png")
            plt.savefig(dist_plot_path)
            plt.close()
            mlflow.log_artifact(dist_plot_path, f"{name}_plots")
            os.remove(dist_plot_path)
    
    
    def __create_training_plots(self, model: CatBoostRegressor, model_name: str, save_folder: Path):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        if hasattr(model, 'evals_result_'):
            evals_result = model.evals_result_
            
            if 'weight' in model_name:
                metric_name = 'MAE'
            else:
                metric_name = 'MultiRMSE'
                
            axes[0].plot(evals_result['learn'][metric_name], label='Train Loss')
            axes[0].plot(evals_result['validation'][metric_name], label='Validation Loss')
            axes[0].set_title(f'{model_name} - Learning Curve ({metric_name})')
            
        
        axes[0].set_xlabel('Iterations')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        sorted_data = sorted([(value, name) for value, name in zip(model.get_feature_importance(), model.feature_names_)], key=lambda x: x[0], reverse=False)
        
        feature_importance = [el[0] for el in sorted_data]
        feature_names = [el[1] for el in sorted_data]
        
        indices = range(len(feature_importance))
        
        axes[1].barh(indices, feature_importance, align='center')
        axes[1].set_yticks(indices)
        axes[1].set_yticklabels(feature_names)
        axes[1].set_title(f'{model_name} - Feature Importance')
        axes[1].set_xlabel('Importance')
        plt.tight_layout()
        plot_path = str(save_folder / f"{model_name}_training_plots.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(plot_path)
        os.remove(plot_path)
    
    
    def run_pipeline(
        self,
        predictor: Predictor,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        weight_max_iterations = None,
        dim_max_iterations = None,
        max_df_size = None
    ):
        """
        Запуск полного pipeline
        """
        save_folder = Path("/app/ml_pipline/mlruns/processed")
        if not save_folder.exists():
            save_folder.mkdir(parents=True)
        
        metadata = {
                "microcat_categories": len(predictor.microcat_df),
        }
        
        
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run() as run:
            logger.info(f"Starting ML pipeline with run {run.info.run_id}...")
            
            if max_df_size is not None:
                _train_data = train_data.head(max_df_size)
                _test_data = test_data.head(max_df_size)
            else:
                _train_data = train_data
                _test_data = test_data
                
            
            train_df, test_df = extract_data(
                predictor=predictor,
                train_df=_train_data,
                test_df=_test_data,
            )
            
            
            self.__prepare_extracted_data(metadata, df=train_df, name='train', save_folder=save_folder)
            self.__prepare_extracted_data(metadata, df=test_df,  name='test',  save_folder=save_folder)
            
            mlflow.log_metrics({
                "train_size": metadata.get("train_samples", 0),
                "test_size": metadata.get("test_samples", 0),
                "microcat_categories": metadata.get("microcat_categories", 0),
            })
                
            
            # TRAIN STAGE 
            weight_model, dim_model = train_models(
                prepared_data=train_df,
                weight_max_iterations=weight_max_iterations,
                dim_max_iterations=dim_max_iterations,
            )
            
            
            mlflow.catboost.log_model(
                weight_model,
                artifact_path="model_weight",
                registered_model_name="weight_model_trained"
            )
            
            mlflow.catboost.log_model(
                dim_model,
                artifact_path="model_dim",
                registered_model_name="dimensions_model_trained"
            )
            
            self.__create_training_plots(model=weight_model, model_name="weight_model", save_folder=save_folder)
            self.__create_training_plots(model=dim_model, model_name="dimensions_model", save_folder=save_folder)
            
            
            # INFERENCE
            
            metrics = inference_models(
                test_data=test_df, 
                weight_model=weight_model,
                dim_model=dim_model
            )
            
            metadata_path =  str(save_folder / "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            mlflow.log_artifact(metadata_path, "extract_data")  
            
            mlflow.log_metrics(metrics)
        
        runs = self.client.search_runs(
            experiment_ids=[mlflow.get_experiment_by_name(self.experiment_name).experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        latest_run = runs[0]
        model_uris = {
            "weight_model": f"runs:/{latest_run.info.run_id}/weight_model",
            "dim_model": f"runs:/{latest_run.info.run_id}/dimensions_model"
        }
     
        return {
            "experiment_name": self.experiment_name,
            "experiment_id": mlflow.get_experiment_by_name(self.experiment_name).experiment_id,
            "run_id": runs[0].info.run_id,
            "model_uris": model_uris
        }

def run_pipeline_from_cli():
    parser = argparse.ArgumentParser(description="Run ML Pipeline")
    parser.add_argument("--train-data-path", type=str, default="data/train_df.parquet", 
                       help="Path to training data parquet")
    parser.add_argument("--test-data-path", type=str, default="data/test_df.parquet", 
                       help="Path to test data parquet")
    parser.add_argument("--microcat-path", type=str, default="data/microcat.json",
                       help="Path to microcat JSON")
    parser.add_argument("--experiment-name", type=str, default=None,
                       help="MLFlow experiment name")
    parser.add_argument("--max-train-iterations", type=int, default=None,
                       help="Max train iterations")
    parser.add_argument("--max-df-size", type=int, default=None,
                       help="Max train and test length")
    
    args = parser.parse_args()
    
    pipeline = MLPipeline(experiment_name=args.experiment_name)
    predictor = Predictor(microcat_json_path=args.microcat_path)
    
    max_df_size = args.max_df_size
    max_iterations = args.max_train_iterations
    
    print(f"START RUNNING PIPELINE. PARAMS:")
    print(f"train_data_path: {args.train_data_path} \ntest_data_path: {args.test_data_path} \nmicrocat_path: {args.microcat_path} \nexperiment_name: {args.experiment_name} \nmax_train_iterations: {max_iterations} \nmax_df_size: {max_df_size}")
    
    result = pipeline.run_pipeline(
        predictor=predictor,
        train_data=pd.read_parquet(args.train_data_path),
        test_data=pd.read_parquet(args.test_data_path),
        weight_max_iterations=max_iterations,
        dim_max_iterations=max_iterations,
        max_df_size=max_df_size
    )
    
    
    print("Pipeline completed!")
    print(f"MLFlow UI: {MLFlowConfig.from_env().tracking_uri}")
    print(f"Experiment ID: {result['experiment_id']}")
    print(f"Run ID: {result['run_id']}")
    print(f"Model URIs: {result['model_uris']}")

if __name__ == "__main__":
    run_pipeline_from_cli()