#!/usr/bin/env python3
"""
Demo script for Epitope Prediction with 3D Visualization
Shows the integration of BepiPred-3.0 and DiscoTope-3.0 with molecular visualization
"""

from bepipred_predictor import BepiPredPredictor
from discotope_predictor import DiscoTopePredictor
from data_retrieval import get_uniprot_sequence, get_structure_content
import time

def demo_epitope_prediction():
    """Demonstration of epitope prediction pipeline"""
    
    print("=" * 60)
    print("🧪 EPITOPE PREDICTION DEMO")
    print("BepiPred-3.0 (Sequence) + DiscoTope-3.0 (Structure)")
    print("=" * 60)
    
    # Test protein
    uniprot_id = "P04637"  # p53
    print(f"\n📋 Testing protein: {uniprot_id} (p53 tumor suppressor)")
    
    # 1. Sequence retrieval
    print("\n🔍 Step 1: Retrieving protein sequence...")
    sequence = get_uniprot_sequence(uniprot_id)
    if sequence:
        print(f"✅ Sequence retrieved: {len(sequence)} amino acids")
        print(f"   First 50 residues: {sequence[:50]}...")
    else:
        print("❌ Failed to retrieve sequence")
        return
    
    # 2. Structure retrieval
    print("\n🏗️  Step 2: Retrieving AlphaFold structure...")
    structure = get_structure_content(uniprot_id, 'alphafold')
    if structure:
        print(f"✅ AlphaFold structure retrieved: {len(structure):,} characters")
    else:
        print("❌ Failed to retrieve structure")
        return
    
    # 3. BepiPred prediction
    print("\n🔬 Step 3: Running BepiPred-3.0 (sequence-based)...")
    try:
        bp_predictor = BepiPredPredictor()
        bp_results = bp_predictor.predict_epitopes(sequence, uniprot_id)
        
        bp_epitopes = [r for r in bp_results if r[3] == "Epitope"]
        print(f"✅ BepiPred-3.0 completed:")
        print(f"   - Total residues: {len(bp_results)}")
        print(f"   - Predicted epitopes: {len(bp_epitopes)}")
        print(f"   - Epitope percentage: {len(bp_epitopes)/len(bp_results)*100:.1f}%")
        
        if bp_epitopes:
            print(f"   - Top 3 epitopes:")
            for i, (pos, aa, score, pred) in enumerate(bp_epitopes[:3]):
                print(f"     {i+1}. Position {pos}: {aa} (score: {score:.3f})")
                
    except Exception as e:
        print(f"❌ BepiPred-3.0 failed: {e}")
        bp_results = None
    
    # 4. DiscoTope prediction
    print("\n🧬 Step 4: Running DiscoTope-3.0 (structure-based)...")
    try:
        dt_predictor = DiscoTopePredictor()
        
        # Use a lower threshold for demo
        dt_results = dt_predictor.predict_epitopes(structure, 'alphafold', threshold=0.5)
        
        dt_epitopes = [r for r in dt_results if r[6] == 1]
        print(f"✅ DiscoTope-3.0 completed:")
        print(f"   - Total residues: {len(dt_results)}")
        print(f"   - Predicted epitopes: {len(dt_epitopes)}")
        print(f"   - Epitope percentage: {len(dt_epitopes)/len(dt_results)*100:.1f}%")
        
        if dt_epitopes:
            print(f"   - Top 3 epitopes:")
            for i, r in enumerate(dt_epitopes[:3]):
                print(f"     {i+1}. Chain {r[1]}, Position {r[2]}: {r[3]} (score: {r[5]:.3f})")
                
    except Exception as e:
        print(f"❌ DiscoTope-3.0 failed: {e}")
        dt_results = None
    
    # 5. Summary
    print("\n📊 Summary:")
    print(f"   - Protein: {uniprot_id} ({len(sequence)} residues)")
    print(f"   - BepiPred epitopes: {len(bp_epitopes) if bp_results else 0}")
    print(f"   - DiscoTope epitopes: {len(dt_epitopes) if dt_results else 0}")
    
    # 6. Next steps
    print("\n🚀 Next Steps:")
    print("   1. Open http://localhost:8502 for 3D visualization")
    print("   2. Enter 'P04637' in the UniProt ID field")
    print("   3. Adjust thresholds as needed")
    print("   4. Click 'Run Epitope Prediction'")
    print("   5. View interactive 3D structure with epitopes highlighted")
    
    print("\n" + "=" * 60)
    print("✨ Demo completed! Check the Streamlit app for visualization.")
    print("=" * 60)

if __name__ == "__main__":
    demo_epitope_prediction()
