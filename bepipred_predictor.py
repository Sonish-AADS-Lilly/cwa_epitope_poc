import tempfile
import os
from pathlib import Path
from typing import List, Tuple
import sys

sys.path.append(str(Path(__file__).parent / "BepiPred-3.0" / "pip" / "Version12_7" / "BP3"))

try:
    from bp3 import bepipred3
    BEPIPRED_AVAILABLE = True
except ImportError:
    print("Warning: BepiPred-3.0 bp3 module not available")
    BEPIPRED_AVAILABLE = False

class BepiPredPredictor:
    def __init__(self):
        self.available = BEPIPRED_AVAILABLE
    
    def predict_epitopes(self, sequence: str, uniprot_id: str = "protein", threshold: float = 0.1512) -> List[Tuple[int, str, float, str]]:
        if not self.available:
            raise RuntimeError("BepiPred-3.0 is not available. Please install the bp3 package.")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                fasta_file = temp_path / "input.fasta"
                esm_dir = temp_path / "esm_encodings"
                
                with open(fasta_file, 'w') as f:
                    f.write(f">{uniprot_id}\n{sequence}\n")
                
                antigens = bepipred3.Antigens(fasta_file, esm_dir, add_seq_len=True)
                predictor = bepipred3.BP3EnsemblePredict(antigens)
                predictor.run_bp3_ensemble()
                
                results = []
                # Get predictions from antigens object
                if not antigens.ensemble_probs or len(antigens.ensemble_probs) == 0:
                    raise RuntimeError("No predictions available")
                
                # Get predictions for the first (and only) sequence
                model_predictions = antigens.ensemble_probs[0]  # List of tensors from different models
                
                # Average the ensemble predictions
                import torch
                stacked_predictions = torch.stack(model_predictions)
                averaged_scores = torch.mean(stacked_predictions, dim=0)
                
                # Convert to list for processing
                scores = averaged_scores.tolist()
                
                for i, (residue, score) in enumerate(zip(sequence, scores)):
                    prediction = "Epitope" if score >= threshold else "Non-Epitope"
                    results.append((i + 1, residue, float(score), prediction))
                
                return results
                
        except Exception as e:
            raise RuntimeError(f"BepiPred prediction failed: {e}")
    
    def predict_with_variable_threshold(self, sequence: str, uniprot_id: str = "protein", threshold: float = 0.1512) -> List[Tuple[int, str, float, str]]:
        return self.predict_epitopes(sequence, uniprot_id, threshold)
