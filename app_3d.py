import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from bepipred_predictor import BepiPredPredictor
from discotope_predictor import DiscoTopePredictor
from data_retrieval import get_uniprot_sequence, get_structure_content
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
from typing import List, Tuple
import stmol
import py3Dmol

st.set_page_config(
    page_title="Epitope Prediction PoC",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #3498db;
        padding-left: 1rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .epitope-highlight {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_3d_structure_viewer(structure_content: str, epitope_results: List[Tuple], title: str = "Protein Structure"):
    """Create 3D molecular viewer with epitopes highlighted"""
    
    if not structure_content or not epitope_results:
        return None
    
    # Create py3Dmol viewer
    view = py3Dmol.view(width=800, height=600)
    view.addModel(structure_content, 'pdb')
    
    # Style the protein
    view.setStyle({'cartoon': {'color': 'lightgray'}})
    
    # Highlight epitopes in red
    epitope_residues = []
    for result in epitope_results:
        if len(result) >= 7 and result[6] == 1:  # prediction == 1 (epitope)
            chain = result[1] if len(result) > 1 else 'A'
            res_num = result[2] if len(result) > 2 else 1
            epitope_residues.append({'chain': chain, 'resi': res_num})
    
    if epitope_residues:
        for residue in epitope_residues:
            view.addStyle({'chain': residue['chain'], 'resi': residue['resi']}, 
                         {'stick': {'colorscheme': 'redCarbon'}, 'sphere': {'color': 'red', 'opacity': 0.8}})
    
    view.zoomTo()
    view.spin(True)
    
    return view

def visualize_epitope_predictions(sequence: str, bp_results: List, dt_results: List = None):
    """Create comprehensive visualizations for epitope predictions"""
    
    # BepiPred visualization
    if bp_results:
        st.markdown('<div class="section-header">BepiPred-3.0 (Sequence-based) Results</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Linear epitope plot
            positions = [r[0] for r in bp_results]
            scores = [r[2] for r in bp_results]
            predictions = [1 if r[3] == "Epitope" else 0 for r in bp_results]
            
            fig = go.Figure()
            
            # Add score line
            fig.add_trace(go.Scatter(
                x=positions, y=scores,
                mode='lines+markers',
                name='BepiPred-3.0 Score',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            
            # Highlight epitopes
            epitope_positions = [pos for pos, pred in zip(positions, predictions) if pred == 1]
            epitope_scores = [score for score, pred in zip(scores, predictions) if pred == 1]
            
            if epitope_positions:
                fig.add_trace(go.Scatter(
                    x=epitope_positions, y=epitope_scores,
                    mode='markers',
                    name='Predicted Epitopes',
                    marker=dict(color='red', size=8, symbol='circle')
                ))
            
            # Add threshold line
            fig.add_hline(y=0.1512, line_dash="dash", line_color="red", 
                         annotation_text="Threshold (0.1512)")
            
            fig.update_layout(
                title="BepiPred-3.0 Epitope Predictions Along Sequence",
                xaxis_title="Residue Position",
                yaxis_title="Epitope Score",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Summary statistics
            total_residues = len(bp_results)
            epitope_count = sum(1 for r in bp_results if r[3] == "Epitope")
            avg_score = sum(r[2] for r in bp_results) / total_residues if total_residues > 0 else 0
            max_score = max(r[2] for r in bp_results) if bp_results else 0
            
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Total Residues", total_residues)
            st.metric("Predicted Epitopes", epitope_count)
            st.metric("Epitope Percentage", f"{epitope_count/total_residues*100:.1f}%" if total_residues > 0 else "0%")
            st.metric("Average Score", f"{avg_score:.3f}")
            st.metric("Max Score", f"{max_score:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Sequence visualization with epitopes highlighted
        st.markdown("**Sequence with Predicted Epitopes:**")
        sequence_html = ""
        for i, (pos, aa, score, pred) in enumerate(bp_results):
            if pred == "Epitope":
                sequence_html += f'<span style="background-color: #ffcccc; font-weight: bold; padding: 2px;">{aa}</span>'
            else:
                sequence_html += aa
            if (i + 1) % 60 == 0:
                sequence_html += "<br>"
        
        st.markdown(f'<div style="font-family: monospace; font-size: 14px; line-height: 1.5;">{sequence_html}</div>', 
                   unsafe_allow_html=True)
    
    # DiscoTope visualization
    if dt_results:
        st.markdown('<div class="section-header">DiscoTope-3.0 (Structure-based) Results</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Structure-based epitope plot
            positions = [r[2] for r in dt_results]
            raw_scores = [r[4] for r in dt_results]
            cal_scores = [r[5] for r in dt_results]
            predictions = [r[6] for r in dt_results]
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Raw scores
            fig.add_trace(go.Scatter(
                x=positions, y=raw_scores,
                mode='lines+markers',
                name='Raw Score',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ), secondary_y=False)
            
            # Calibrated scores
            fig.add_trace(go.Scatter(
                x=positions, y=cal_scores,
                mode='lines+markers',
                name='Calibrated Score',
                line=dict(color='green', width=2),
                marker=dict(size=4)
            ), secondary_y=False)
            
            # Highlight epitopes
            epitope_positions = [pos for pos, pred in zip(positions, predictions) if pred == 1]
            epitope_cal_scores = [cal for cal, pred in zip(cal_scores, predictions) if pred == 1]
            
            if epitope_positions:
                fig.add_trace(go.Scatter(
                    x=epitope_positions, y=epitope_cal_scores,
                    mode='markers',
                    name='Predicted Epitopes',
                    marker=dict(color='red', size=10, symbol='circle')
                ), secondary_y=False)
            
            fig.update_layout(
                title="DiscoTope-3.0 Structure-based Epitope Predictions",
                xaxis_title="Residue Position",
                height=400
            )
            fig.update_yaxes(title_text="Epitope Score", secondary_y=False)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Summary statistics
            total_residues = len(dt_results)
            epitope_count = sum(1 for r in dt_results if r[6] == 1)
            avg_raw_score = sum(r[4] for r in dt_results) / total_residues if total_residues > 0 else 0
            avg_cal_score = sum(r[5] for r in dt_results) / total_residues if total_residues > 0 else 0
            
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Total Residues", total_residues)
            st.metric("Predicted Epitopes", epitope_count)
            st.metric("Epitope Percentage", f"{epitope_count/total_residues*100:.1f}%" if total_residues > 0 else "0%")
            st.metric("Avg Raw Score", f"{avg_raw_score:.3f}")
            st.metric("Avg Calibrated Score", f"{avg_cal_score:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)

def run_predictions(uniprot_id: str, sequence: str, structure_content: str, structure_type: str):
    """Run both BepiPred and DiscoTope predictions"""
    
    results = {}
    
    # BepiPred prediction
    with st.spinner("Running BepiPred-3.0 prediction..."):
        try:
            bp_predictor = BepiPredPredictor()
            bp_results = bp_predictor.predict_epitopes(sequence, uniprot_id)
            results['bepipred'] = bp_results
            st.success(f"BepiPred-3.0 completed: {len(bp_results)} residues analyzed")
        except Exception as e:
            st.error(f"BepiPred-3.0 failed: {str(e)}")
            results['bepipred'] = None
    
    # DiscoTope prediction
    if structure_content:
        with st.spinner("Running DiscoTope-3.0 prediction..."):
            try:
                dt_predictor = DiscoTopePredictor()
                dt_results = dt_predictor.predict_epitopes(structure_content, structure_type)
                results['discotope'] = dt_results
                st.success(f"DiscoTope-3.0 completed: {len(dt_results)} residues analyzed")
            except Exception as e:
                st.error(f"DiscoTope-3.0 failed: {str(e)}")
                results['discotope'] = None
    else:
        results['discotope'] = None
    
    return results

def main():
    st.markdown('<div class="main-header">Epitope Prediction Platform</div>', unsafe_allow_html=True)
    st.markdown("BepiPred-3.0 (sequence-based) and DiscoTope-3.0 (structure-based) predictions")
    
    # Sidebar for input
    with st.sidebar:
        st.header("Input Parameters")
        
        # UniProt ID input
        uniprot_id = st.text_input(
            "UniProt ID",
            value="P04637",
            help="Enter a UniProt accession ID (e.g., P04637 for p53)"
        )
        
        # Threshold settings
        st.subheader("Prediction Thresholds")
        bp_threshold = st.slider("BepiPred-3.0 Threshold", 0.0, 1.0, 0.1512, 0.001)
        dt_threshold = st.slider("DiscoTope-3.0 Threshold", 0.0, 2.0, 0.90, 0.01)
        
        # Structure source
        structure_source = st.selectbox(
            "Structure Source",
            ["alphafold", "pdb"],
            help="Choose between AlphaFold predictions or experimental PDB structures"
        )
        
        # Run prediction button
        run_button = st.button("Run Epitope Prediction", type="primary", use_container_width=True)
    
    # Main content area
    if run_button and uniprot_id:
        st.markdown('<div class="section-header">Data Retrieval</div>', unsafe_allow_html=True)
        
        # Get sequence
        with st.spinner("Fetching protein sequence..."):
            sequence = get_uniprot_sequence(uniprot_id)
            if sequence:
                st.success(f"Retrieved sequence: {len(sequence)} amino acids")
                with st.expander("View Sequence"):
                    st.text(sequence)
            else:
                st.error("Failed to retrieve sequence")
                return
        
        # Get structure
        with st.spinner(f"Fetching {structure_source} structure..."):
            structure_content = get_structure_content(uniprot_id, structure_source)
            if structure_content:
                st.success(f"Retrieved {structure_source} structure: {len(structure_content)} characters")
            else:
                st.warning("Failed to retrieve structure - DiscoTope predictions will be skipped")
        
        # Run predictions
        st.markdown('<div class="section-header">Epitope Predictions</div>', unsafe_allow_html=True)
        results = run_predictions(uniprot_id, sequence, structure_content, structure_source)
        
        # Display results
        if results['bepipred'] or results['discotope']:
            visualize_epitope_predictions(sequence, results['bepipred'], results['discotope'])
            
            # 3D Structure Visualization
            if structure_content and results['discotope']:
                st.markdown('<div class="section-header">3D Structure Visualization</div>', unsafe_allow_html=True)
                
                try:
                    view = create_3d_structure_viewer(structure_content, results['discotope'], 
                                                    f"{uniprot_id} - Epitopes Highlighted")
                    if view:
                        st.markdown("**Interactive 3D Structure** (Epitopes in red):")
                        stmol.showmol(view, height=600, width=800)
                    else:
                        st.warning("Could not create 3D visualization")
                except Exception as e:
                    st.error(f"3D visualization failed: {str(e)}")
            
            # Data tables
            st.markdown('<div class="section-header">Detailed Results</div>', unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["BepiPred-3.0 Results", "DiscoTope-3.0 Results"])
            
            with tab1:
                if results['bepipred']:
                    bp_df = pd.DataFrame(results['bepipred'], 
                                       columns=['Position', 'Residue', 'Score', 'Prediction'])
                    epitope_df = bp_df[bp_df['Prediction'] == 'Epitope']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("All Predictions")
                        st.dataframe(bp_df, use_container_width=True)
                    
                    with col2:
                        st.subheader("Predicted Epitopes Only")
                        if not epitope_df.empty:
                            st.dataframe(epitope_df, use_container_width=True)
                        else:
                            st.info("No epitopes predicted with current threshold")
                else:
                    st.info("BepiPred-3.0 results not available")
            
            with tab2:
                if results['discotope']:
                    dt_df = pd.DataFrame(results['discotope'], 
                                       columns=['PDB_ID', 'Chain', 'Position', 'Residue', 
                                              'Raw_Score', 'Calibrated_Score', 'Prediction'])
                    epitope_df = dt_df[dt_df['Prediction'] == 1]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("All Predictions")
                        st.dataframe(dt_df, use_container_width=True)
                    
                    with col2:
                        st.subheader("Predicted Epitopes Only")
                        if not epitope_df.empty:
                            st.dataframe(epitope_df, use_container_width=True)
                        else:
                            st.info("No epitopes predicted with current threshold")
                else:
                    st.info("DiscoTope-3.0 results not available")
        
        else:
            st.error("No predictions were successful")
    
    elif run_button:
        st.warning("Please enter a UniProt ID")

if __name__ == "__main__":
    main()
