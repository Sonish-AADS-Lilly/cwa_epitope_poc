import tempfile
import subprocess
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DISCOTOPE_DIR = Path(__file__).parent / "DiscoTope-3.0"

class DiscoTopePredictor:
    def __init__(self):
        self.discotope_main = DISCOTOPE_DIR / "discotope3" / "main.py"
        self.available = self.discotope_main.exists()
        logger.info(f"DiscoTope available: {self.available}")
        logger.info(f"DiscoTope main script: {self.discotope_main}")
    
    def predict_epitopes(self, structure_content: str, structure_type: str, threshold: float = 0.90) -> List[Tuple[str, str, int, str, float, float, int]]:
        """Use DiscoTope command-line interface for prediction"""
        logger.info(f"Starting DiscoTope CLI prediction with structure_type: {structure_type}")
        
        if not self.available:
            logger.error("DiscoTope-3.0 CLI is not available")
            raise RuntimeError("DiscoTope-3.0 CLI is not available")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pdb_file = temp_path / "structure.pdb"
            output_dir = temp_path / "output"
            output_dir.mkdir()
            
            # Write PDB file
            with open(pdb_file, 'w') as f:
                f.write(structure_content)
            logger.info(f"Written PDB file: {pdb_file}")
            
            # Run DiscoTope CLI
            cmd = [
                sys.executable, str(self.discotope_main),
                "--pdb_or_zip_file", str(pdb_file),
                "--struc_type", structure_type,
                "--out_dir", str(output_dir),
                "--calibrated_score_epi_threshold", str(threshold),
                "--cpu_only"  # Force CPU to avoid GPU issues
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300,  # 5 minute timeout
                cwd=str(DISCOTOPE_DIR)
            )
        except subprocess.TimeoutExpired:
            logger.error("DiscoTope CLI timed out after 5 minutes")
            raise RuntimeError(
                "DiscoTope CLI timed out. This may indicate system resource issues "
                "or problems with the ESM model loading."
            )
            
            if result.returncode != 0:
                logger.error(f"DiscoTope CLI failed with return code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                
                # Provide specific error message for segmentation fault
                if result.returncode == -11:
                    error_msg = (
                        "DiscoTope CLI failed due to segmentation fault (likely ESM-IF1 model issue). "
                        "This is a known issue with the official DiscoTope-3.0 implementation. "
                        "Try using a different protein structure or check system requirements."
                    )
                else:
                    error_msg = f"DiscoTope CLI failed: {result.stderr}"
                
                raise RuntimeError(error_msg)
            
            logger.info("DiscoTope CLI completed successfully")
            logger.info(f"STDOUT: {result.stdout}")
            
            # Parse output CSV files
            csv_files = list(output_dir.glob("*.csv"))
            if not csv_files:
                raise RuntimeError("No CSV output files found")
            
            all_results = []
            for csv_file in csv_files:
                logger.info(f"Reading CSV file: {csv_file}")
                df = pd.read_csv(csv_file)
                
                for _, row in df.iterrows():
                    prediction = 1 if row.get('predicted_epitope', False) else 0
                    all_results.append((
                        str(row['pdb']), 
                        str(row['pdb']).split('_')[-1] if '_' in str(row['pdb']) else 'A',  # chain
                        int(row['res_id']), 
                        str(row['residue']),
                        float(row['DiscoTope-3.0_score']),  # raw score
                        float(row['DiscoTope-3.0_score']),  # calibrated score (same for CLI)
                        prediction
                    ))
            
            logger.info(f"CLI prediction complete: {len(all_results)} residues")
            return all_results
