import tempfile
import pickle
import xgboost as xgb
import numpy as np
from pathlib import Path
from typing import List, Tuple
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).parent / "DiscoTope-3.0"))

try:
    from discotope3.make_dataset import Discotope_Dataset_web
    DISCOTOPE_AVAILABLE = True
    logger.info("DiscoTope-3.0 modules loaded successfully")
except ImportError as e:
    DISCOTOPE_AVAILABLE = False
    logger.warning(f"DiscoTope-3.0 not available: {e}")

class DiscoTopePredictor:
    def __init__(self):
        self.available = DISCOTOPE_AVAILABLE
        self.models_dir = Path(__file__).parent / "DiscoTope-3.0" / "models"
        logger.info(f"DiscoTope available: {self.available}")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Models directory exists: {self.models_dir.exists()}")
    
    def predict_epitopes(self, structure_content: str, structure_type: str, threshold: float = 0.90) -> List[Tuple[str, str, int, str, float, float, int]]:
        logger.info(f"Starting DiscoTope prediction with structure_type: {structure_type}")
        
        if not self.available:
            logger.error("DiscoTope-3.0 is not available")
            raise RuntimeError("DiscoTope-3.0 is not available")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                pdb_file = temp_path / "structure.pdb"
                logger.info(f"Created temporary PDB file: {pdb_file}")
                
                with open(pdb_file, 'w') as f:
                    f.write(structure_content)
                logger.info(f"Written {len(structure_content)} characters to PDB file")
                
                logger.info("Creating DiscoTope dataset...")
                try:
                    import signal
                    import os
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("DiscoTope dataset creation timed out")
                    
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(120)  # 2 minute timeout
                    
                    dataset = Discotope_Dataset_web(str(temp_path), structure_type)
                    signal.alarm(0)  # Cancel timeout
                    logger.info("Dataset created successfully")
                    
                except (TimeoutError, MemoryError, OSError, RuntimeError) as dataset_error:
                    logger.error(f"DiscoTope dataset creation failed: {dataset_error}")
                    raise RuntimeError(f"DiscoTope prediction unavailable due to system limitations: {dataset_error}")
                except Exception as dataset_error:
                    logger.error(f"Unexpected error creating dataset: {dataset_error}")
                    raise RuntimeError(f"Failed to create DiscoTope dataset: {dataset_error}")
                
                if not hasattr(dataset, 'df') or dataset.df.empty:
                    logger.error("Dataset is empty or invalid")
                    raise RuntimeError("Failed to process structure")
                
                logger.info(f"Dataset contains {len(dataset.df)} residues")
                features = dataset.df[dataset.feature_list].values
                logger.info(f"Features shape: {features.shape}")
                
                model_predictions = []
                model_count = 0
                for i in range(1, 101):
                    model_file = self.models_dir / f"XGB_{i}_of_100.json"
                    if model_file.exists():
                        model = xgb.XGBClassifier()
                        model.load_model(str(model_file))
                        pred_proba = model.predict_proba(features)[:, 1]
                        model_predictions.append(pred_proba)
                        model_count += 1
                
                logger.info(f"Loaded {model_count} XGBoost models")
                
                if not model_predictions:
                    logger.error("No XGBoost models found")
                    raise RuntimeError("No XGBoost models found")
                
                raw_scores = np.mean(model_predictions, axis=0)
                logger.info(f"Raw scores calculated, range: {raw_scores.min():.3f} - {raw_scores.max():.3f}")
                
                gam_len_file = self.models_dir / "gam_len_to_mean.pkl"
                gam_surface_file = self.models_dir / "gam_surface_to_std.pkl"
                
                if gam_len_file.exists() and gam_surface_file.exists():
                    logger.info("Applying GAM calibration...")
                    with open(gam_len_file, 'rb') as f:
                        gam_len = pickle.load(f)
                    with open(gam_surface_file, 'rb') as f:
                        gam_surface = pickle.load(f)
                    
                    chain_lengths = dataset.df['chain_len'].values
                    surface_accessibility = dataset.df['total_surf_acc'].values
                    
                    length_corrections = gam_len.predict(chain_lengths.reshape(-1, 1))
                    surface_corrections = gam_surface.predict(surface_accessibility.reshape(-1, 1))
                    
                    calibrated_scores = raw_scores - length_corrections + surface_corrections
                    logger.info(f"Calibrated scores range: {calibrated_scores.min():.3f} - {calibrated_scores.max():.3f}")
                else:
                    logger.warning("GAM calibration files not found, using raw scores")
                    calibrated_scores = raw_scores
                
                results = []
                epitope_count = 0
                for idx, row in dataset.df.iterrows():
                    prediction = 1 if calibrated_scores[idx] >= threshold else 0
                    if prediction == 1:
                        epitope_count += 1
                    results.append((
                        row['pdb_id'], row['chain'], int(row['pos']), row['aa'],
                        float(raw_scores[idx]), float(calibrated_scores[idx]), prediction
                    ))
                
                logger.info(f"Prediction complete: {epitope_count} epitopes out of {len(results)} residues")
                return results
                
        except Exception as e:
            logger.error(f"DiscoTope prediction failed: {e}")
            raise RuntimeError(f"DiscoTope prediction failed: {e}")
