#!/usr/bin/env python3

import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_discotope_full():
    """Complete test of DiscoTope functionality"""
    
    logger.info("=== DiscoTope-3.0 Full Test ===")
    
    # Test 1: Import and initialization
    try:
        from discotope_predictor import DiscoTopePredictor
        predictor = DiscoTopePredictor()
        logger.info(f"✓ DiscoTope predictor initialized. Available: {predictor.available}")
    except Exception as e:
        logger.error(f"✗ Failed to initialize DiscoTope: {e}")
        return False
    
    if not predictor.available:
        logger.error("✗ DiscoTope not available - cannot proceed with test")
        return False
    
    # Test 2: Check models directory
    models_dir = predictor.models_dir
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Models directory exists: {models_dir.exists()}")
    
    if models_dir.exists():
        xgb_models = list(models_dir.glob("XGB_*_of_100.json"))
        logger.info(f"Found {len(xgb_models)} XGBoost models")
        
        gam_len_file = models_dir / "gam_len_to_mean.pkl"
        gam_surface_file = models_dir / "gam_surface_to_std.pkl"
        logger.info(f"GAM length file exists: {gam_len_file.exists()}")
        logger.info(f"GAM surface file exists: {gam_surface_file.exists()}")
    
    # Test 3: Test with sample PDB content
    sample_pdb = """HEADER    SAMPLE PDB                              01-JAN-70   TEST
ATOM      1  N   MET A   1      20.154  16.967  23.421  1.00 25.00           N  
ATOM      2  CA  MET A   1      20.154  16.967  24.421  1.00 25.00           C  
ATOM      3  C   MET A   1      21.154  16.967  24.421  1.00 25.00           C  
ATOM      4  O   MET A   1      22.154  16.967  24.421  1.00 25.00           O  
ATOM      5  CB  MET A   1      19.154  16.967  24.421  1.00 25.00           C  
ATOM      6  N   ALA A   2      21.154  16.967  25.421  1.00 25.00           N  
ATOM      7  CA  ALA A   2      21.154  16.967  26.421  1.00 25.00           C  
ATOM      8  C   ALA A   2      22.154  16.967  26.421  1.00 25.00           C  
ATOM      9  O   ALA A   2      23.154  16.967  26.421  1.00 25.00           O  
ATOM     10  CB  ALA A   2      20.154  16.967  26.421  1.00 25.00           C  
END
"""
    
    try:
        logger.info("Testing with sample PDB structure...")
        results = predictor.predict_epitopes(sample_pdb, "solved", 0.5)
        logger.info(f"✓ DiscoTope prediction successful: {len(results)} results")
        
        if results:
            logger.info("Sample results:")
            for i, result in enumerate(results[:3]):  # Show first 3 results
                pdb_id, chain, pos, aa, raw_score, cal_score, pred = result
                logger.info(f"  {i+1}: {chain}{pos}{aa} -> Raw: {raw_score:.3f}, Cal: {cal_score:.3f}, Pred: {pred}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ DiscoTope prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_discotope_with_real_data():
    """Test with actual UniProt data"""
    
    logger.info("=== Testing with Real UniProt Data ===")
    
    try:
        from data_retrieval import get_uniprot_sequence, get_structure_content
        from discotope_predictor import DiscoTopePredictor
        
        # Test with a small protein
        uniprot_id = "P00779"  # Cytochrome c
        logger.info(f"Testing with UniProt ID: {uniprot_id}")
        
        sequence = get_uniprot_sequence(uniprot_id)
        logger.info(f"Retrieved sequence length: {len(sequence)}")
        
        structure_content, structure_type = get_structure_content(uniprot_id)
        logger.info(f"Structure type: {structure_type}")
        logger.info(f"Structure content length: {len(structure_content) if structure_content else 0}")
        
        if structure_content:
            predictor = DiscoTopePredictor()
            if predictor.available:
                results = predictor.predict_epitopes(structure_content, structure_type, 0.9)
                logger.info(f"✓ Real data prediction successful: {len(results)} results")
                
                epitope_count = sum(1 for r in results if r[6] == 1)
                logger.info(f"Epitopes found: {epitope_count}/{len(results)}")
                
                return True
            else:
                logger.warning("DiscoTope not available for real data test")
                return False
        else:
            logger.warning("No structure available for real data test")
            return False
            
    except Exception as e:
        logger.error(f"✗ Real data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting DiscoTope comprehensive test...")
    
    # Run basic test
    basic_success = test_discotope_full()
    
    # Run real data test if basic test passes
    real_data_success = False
    if basic_success:
        real_data_success = test_discotope_with_real_data()
    
    # Summary
    logger.info("=== Test Summary ===")
    logger.info(f"Basic functionality: {'✓ PASS' if basic_success else '✗ FAIL'}")
    logger.info(f"Real data test: {'✓ PASS' if real_data_success else '✗ FAIL'}")
    
    if basic_success and real_data_success:
        logger.info("🎉 All DiscoTope tests PASSED!")
        sys.exit(0)
    else:
        logger.error("❌ Some DiscoTope tests FAILED!")
        sys.exit(1)
