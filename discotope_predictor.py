import tempfile
import pickle
import xgboost as xgb
import numpy as np
from pathlib import Path
from typing import List, Tuple
import sys

sys.path.append(str(Path(__file__).parent / "DiscoTope-3.0"))

try:
    from discotope3.make_dataset import Discotope_Dataset_web
    DISCOTOPE_AVAILABLE = True
except ImportError:
    DISCOTOPE_AVAILABLE = False

class DiscoTopePredictor:
    def __init__(self):
        self.available = DISCOTOPE_AVAILABLE
        self.models_dir = Path(__file__).parent / "DiscoTope-3.0" / "models"
    
    def predict_epitopes(self, structure_content: str, structure_type: str, threshold: float = 0.90) -> List[Tuple[str, str, int, str, float, float, int]]:
        if not self.available:
            raise RuntimeError("DiscoTope-3.0 is not available")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                pdb_file = temp_path / "structure.pdb"
                
                with open(pdb_file, 'w') as f:
                    f.write(structure_content)
                
                dataset = Discotope_Dataset_web(str(pdb_file))
                
                if not hasattr(dataset, 'df') or dataset.df.empty:
                    raise RuntimeError("Failed to process structure")
                
                features = dataset.df[dataset.feature_list].values
                
                model_predictions = []
                for i in range(1, 101):
                    model_file = self.models_dir / f"XGB_{i}_of_100.json"
                    if model_file.exists():
                        model = xgb.XGBClassifier()
                        model.load_model(str(model_file))
                        pred_proba = model.predict_proba(features)[:, 1]
                        model_predictions.append(pred_proba)
                
                if not model_predictions:
                    raise RuntimeError("No XGBoost models found")
                
                raw_scores = np.mean(model_predictions, axis=0)
                
                gam_len_file = self.models_dir / "gam_len_to_mean.pkl"
                gam_surface_file = self.models_dir / "gam_surface_to_std.pkl"
                
                if gam_len_file.exists() and gam_surface_file.exists():
                    with open(gam_len_file, 'rb') as f:
                        gam_len = pickle.load(f)
                    with open(gam_surface_file, 'rb') as f:
                        gam_surface = pickle.load(f)
                    
                    chain_lengths = dataset.df['chain_len'].values
                    surface_accessibility = dataset.df['total_surf_acc'].values
                    
                    length_corrections = gam_len.predict(chain_lengths.reshape(-1, 1))
                    surface_corrections = gam_surface.predict(surface_accessibility.reshape(-1, 1))
                    
                    calibrated_scores = raw_scores - length_corrections + surface_corrections
                else:
                    calibrated_scores = raw_scores
                
                results = []
                for idx, row in dataset.df.iterrows():
                    prediction = 1 if calibrated_scores[idx] >= threshold else 0
                    results.append((
                        row['pdb_id'], row['chain'], int(row['pos']), row['aa'],
                        float(raw_scores[idx]), float(calibrated_scores[idx]), prediction
                    ))
                
                return results
                
        except Exception as e:
            raise RuntimeError(f"DiscoTope prediction failed: {e}")
