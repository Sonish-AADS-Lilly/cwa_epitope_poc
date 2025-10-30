import tempfile
import os
import sys
import pickle
import glob
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent / "DiscoTope-3.0"))

try:
    import xgboost as xgb
    import biotite.structure as struc
    import biotite.structure.io as strucio
    from discotope3.make_dataset import Discotope_Dataset_web
    DISCOTOPE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DiscoTope-3.0 dependencies not available: {e}")
    DISCOTOPE_AVAILABLE = False

class DiscoTopePredictor:
    def __init__(self):
        self.available = DISCOTOPE_AVAILABLE
        self.models = None
        self.gam_len_to_mean = None
        self.gam_surface_to_std = None
        
        if self.available:
            self._load_models()
    
    def _load_models(self):
        try:
            models_dir = Path(__file__).parent / "DiscoTope-3.0" / "models"
            if not models_dir.exists():
                print("DiscoTope-3.0 models directory not found")
                self.available = False
                return
            
            model_files = list(models_dir.glob("XGB_*_of_*.json"))
            if len(model_files) < 100:
                print(f"Loading available XGBoost models: {len(model_files)}/100")
            
            self.models = []
            for fp in model_files:
                model = xgb.XGBClassifier()
                model.load_model(str(fp))
                self.models.append(model)
            
            gam_len_path = models_dir / "gam_len_to_mean.pkl"
            gam_surface_path = models_dir / "gam_surface_to_std.pkl"
            
            if gam_len_path.exists() and gam_surface_path.exists():
                with open(gam_len_path, 'rb') as f:
                    self.gam_len_to_mean = pickle.load(f)
                with open(gam_surface_path, 'rb') as f:
                    self.gam_surface_to_std = pickle.load(f)
            else:
                print("Warning: GAM calibration models not found")
                
        except Exception as e:
            print(f"Failed to load DiscoTope models: {e}")
            self.available = False
    
    def predict_epitopes(self, pdb_content: str, structure_type: str = "alphafold", 
                        confidence_threshold: float = 0.90) -> List[Tuple[str, str, int, str, float, float, int]]:
        if not self.available or not self.models:
            raise RuntimeError("DiscoTope-3.0 is not available. Please install required dependencies and models.")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                pdb_file = temp_path / "structure.pdb"
                
                with open(pdb_file, 'w') as f:
                    f.write(pdb_content)
                
                dataset = Discotope_Dataset_web(
                    str(temp_path),
                    structure_type=structure_type,
                    check_existing_embeddings=False,
                    save_embeddings=False,
                    cpu_only=True,
                    max_gpu_pdb_length=1000,
                    verbose=0
                )
                
                if len(dataset) == 0:
                    raise RuntimeError("No valid PDB structures found")
                
                X_all = np.concatenate([
                    dataset[i]["X_arr"] for i in range(len(dataset))
                    if dataset[i]["X_arr"] is not False
                ])
                
                y_pred = self._predict_with_models(X_all)
                
                df_all = pd.concat([
                    dataset[i]["df_stats"] for i in range(len(dataset))
                    if dataset[i]["X_arr"] is not False
                ])
                
                df_all["DiscoTope-3.0_score"] = y_pred
                
                results = []
                start = 0
                for i in range(len(dataset)):
                    if dataset[i]["X_arr"] is False:
                        continue
                    
                    sample = dataset[i]
                    length = len(sample["X_arr"])
                    end = start + length
                    
                    df_subset = df_all.iloc[start:end].copy()
                    
                    if self.gam_len_to_mean is not None and self.gam_surface_to_std is not None:
                        calibrated_scores = self._normalize_scores(df_subset)
                        df_subset["calibrated_score"] = calibrated_scores
                    else:
                        df_subset["calibrated_score"] = df_subset["DiscoTope-3.0_score"] * 1.15
                    
                    for idx, (_, row) in enumerate(df_subset.iterrows()):
                        score = float(row["DiscoTope-3.0_score"])
                        calibrated_score = float(row["calibrated_score"])
                        prediction = 1 if calibrated_score >= confidence_threshold else 0
                        
                        results.append((
                            sample["pdb_id"],
                            row["chain"],
                            int(row["res_id"]),
                            row["residue"],
                            score,
                            calibrated_score,
                            prediction
                        ))
                    
                    start = end
                
                return results
                
        except Exception as e:
            raise RuntimeError(f"DiscoTope prediction failed: {e}")
    
    def _predict_with_models(self, X: np.ndarray) -> np.ndarray:
        y_pred = np.zeros(len(X))
        for model in self.models:
            pred = model.predict_proba(X)
            if len(pred.shape) > 1:
                pred = pred[:, 1]
            y_pred += pred
        return y_pred / len(self.models)
    
    def _normalize_scores(self, df: pd.DataFrame) -> np.ndarray:
        scores = df["DiscoTope-3.0_score"].astype(float).values
        length = int(df["length"].iloc[0])
        
        surface_scores = df[df["rsa"].astype(float) >= 0.20]
        if len(surface_scores) > 0:
            mean_surface_score = surface_scores["rsa"].astype(float).mean()
        else:
            mean_surface_score = 0.5
        
        try:
            u = self.gam_len_to_mean.predict(np.array([length]))[0]
            std = self.gam_surface_to_std.predict(np.array([mean_surface_score]))[0]
            calibrated_scores = (scores - u) / std
            return calibrated_scores
        except:
            return scores * 1.15
