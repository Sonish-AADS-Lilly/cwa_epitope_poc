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
import py3Dmol

try:
    import stmol
    STMOL_AVAILABLE = True
except ImportError:
    STMOL_AVAILABLE = False

st.set_page_config(
    page_title="Epitope Prediction PoC",
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

def create_3d_structure_viewer(pdb_content: str, epitope_positions: List[int], protein_id: str = "protein"):
    try:
        view = py3Dmol.view(width=800, height=600)
        view.addModel(pdb_content, 'pdb')
        
        view.setStyle({'cartoon': {'color': 'lightblue'}})
        
        if epitope_positions:
            for pos in epitope_positions:
                view.addStyle({'resi': pos}, {'sphere': {'color': 'red', 'radius': 1.5}})
        
        view.zoomTo()
        return view
    except Exception as e:
        st.error(f"Error creating 3D viewer: {str(e)}")
        return None

def visualize_epitope_predictions(sequence: str, bp_results: List, dt_results: List = None, dt_fallback: bool = False):
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
            
            # Create title based on fallback status
            title = "DiscoTope-3.0 Structure-based Epitope Predictions"
            if dt_fallback:
                title += " (Fallback Mode)"
            
            fig.update_layout(
                title=title,
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

def extract_peptides(sequence: str, epitope_positions: List[int], center_position: int = 4, min_length: int = 3) -> List[dict]:
    peptides = []
    epitope_positions = sorted(epitope_positions)
    
    groups = []
    current_group = [epitope_positions[0]] if epitope_positions else []
    
    for i in range(1, len(epitope_positions)):
        if epitope_positions[i] - epitope_positions[i-1] == 1:
            current_group.append(epitope_positions[i])
        else:
            if len(current_group) >= min_length:
                groups.append(current_group)
            current_group = [epitope_positions[i]]
    
    if len(current_group) >= min_length:
        groups.append(current_group)
    
    for group in groups:
        start_pos = max(0, group[0] - 1 - center_position)
        end_pos = min(len(sequence), group[-1] + center_position)
        
        peptide_seq = sequence[start_pos:end_pos]
        core_start = group[0] - 1 - start_pos
        core_end = group[-1] - start_pos
        
        peptides.append({
            'Peptide': peptide_seq,
            'Start': start_pos + 1,
            'End': end_pos,
            'Length': len(peptide_seq),
            'Core_Epitope': sequence[group[0]-1:group[-1]],
            'Core_Start': core_start + 1,
            'Core_End': core_end,
            'Positions': f"{group[0]}-{group[-1]}"
        })
    
    return peptides

def recalculate_epitopes(sequence: str, bp_results: List, dt_results: List, 
                        bp_threshold: float, dt_threshold: float, 
                        center_position: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict], List[dict]]:
    results = {'bepipred': None, 'discotope': None, 'bepipred_peptides': [], 'discotope_peptides': []}
    
    if bp_results:
        # Recalculate BepiPred predictions
        bp_recalc = []
        bp_epitope_positions = []
        
        for pos, residue, score, _ in bp_results:
            prediction = "Epitope" if score >= bp_threshold else "Non-Epitope"
            bp_recalc.append((pos, residue, score, prediction))
            if prediction == "Epitope":
                bp_epitope_positions.append(pos)
        
        results['bepipred'] = bp_recalc
        results['bepipred_peptides'] = extract_peptides(sequence, bp_epitope_positions)
    
    if dt_results:
        # Recalculate DiscoTope predictions
        dt_recalc = []
        dt_epitope_positions = []
        
        for pdb_id, chain, pos, residue, raw_score, calib_score, _ in dt_results:
            prediction = 1 if calib_score >= dt_threshold else 0
            dt_recalc.append((pdb_id, chain, pos, residue, raw_score, calib_score, prediction))
            if prediction == 1:
                dt_epitope_positions.append(pos)
        
        results['discotope'] = dt_recalc
        results['discotope_peptides'] = extract_peptides(sequence, dt_epitope_positions)
    
    return results

def run_predictions(uniprot_id: str, sequence: str, structure_content: str, structure_type: str):
    
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
                results['discotope_fallback'] = getattr(dt_predictor, 'using_fallback', False)
                
                # Check if fallback was used and display appropriate message
                if hasattr(dt_predictor, 'using_fallback') and dt_predictor.using_fallback:
                    st.warning("🔄 **DiscoTope-3.0 Fallback Mode Active**")
                    st.info(
                        f"Official DiscoTope-3.0 encountered technical issues and switched to fallback mode. "
                        f"Results are generated using structure-based heuristics. "
                        f"**{len(dt_results)} residues analyzed**"
                    )
                else:
                    st.success(f"✅ **DiscoTope-3.0 Official Model**: {len(dt_results)} residues analyzed")
            except Exception as e:
                error_msg = str(e)
                st.error(f"DiscoTope-3.0 failed: {error_msg}")
                results['discotope'] = None
                results['discotope_fallback'] = False
    else:
        results['discotope'] = None
        results['discotope_fallback'] = False
    
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
            visualize_epitope_predictions(sequence, results['bepipred'], results['discotope'], 
                                         results.get('discotope_fallback', False))
            
            # 3D Structure Visualization
            if structure_content and results['discotope']:
                st.markdown('<div class="section-header">3D Structure Visualization</div>', unsafe_allow_html=True)
                
                try:
                    epitope_positions = [r[2] for r in results['discotope'] if r[6] == 1]
                    view = create_3d_structure_viewer(structure_content, epitope_positions, 
                                                    f"{uniprot_id} - Epitopes Highlighted")
                    if view and STMOL_AVAILABLE:
                        st.markdown("**Interactive 3D Structure** (Epitopes in red):")
                        stmol.showmol(view, height=600, width=800)
                    elif view:
                        st.warning("3D visualization requires stmol package. Install with: pip install stmol")
                        st.info(f"Structure loaded with {len(epitope_positions)} epitope positions highlighted")
                    else:
                        st.warning("Could not create 3D visualization")
                except Exception as e:
                    st.error(f"3D visualization failed: {str(e)}")
            
            # Data tables with interactive recalculation
            st.markdown('<div class="section-header">Detailed Results</div>', unsafe_allow_html=True)
            
            # Interactive parameter controls
            with st.expander("🔧 Recalculate with Different Parameters", expanded=False):
                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                
                with col1:
                    new_bp_threshold = st.number_input(
                        "BepiPred Threshold:", 
                        min_value=0.0, max_value=1.0, value=0.1512, step=0.01,
                        help="Threshold for BepiPred epitope prediction"
                    )
                
                with col2:
                    new_dt_threshold = st.number_input(
                        "DiscoTope Threshold:", 
                        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                        help="Threshold for DiscoTope epitope prediction"
                    )
                
                with col3:
                    center_position = st.number_input(
                        "Center Position:", 
                        min_value=0, max_value=10, value=4, step=1,
                        help="Number of residues to extend around epitope core"
                    )
                
                with col4:
                    if st.button("🔄 Recalculate", use_container_width=True):
                        # Recalculate with new parameters
                        recalc_results = recalculate_epitopes(
                            sequence, results['bepipred'], results['discotope'],
                            new_bp_threshold, new_dt_threshold
                        )
                        # Update session state with recalculated results
                        st.session_state.recalc_results = recalc_results
                        st.session_state.center_position = center_position
                        st.rerun()
            
            # Use recalculated results if available
            display_results = getattr(st.session_state, 'recalc_results', results)
            center_pos = getattr(st.session_state, 'center_position', 4)
            
            # Generate peptides if not already calculated
            if 'bepipred_peptides' not in display_results:
                bp_epitope_positions = []
                dt_epitope_positions = []
                
                if display_results['bepipred']:
                    bp_epitope_positions = [pos for pos, _, _, pred in display_results['bepipred'] if pred == 'Epitope']
                
                if display_results['discotope']:
                    dt_epitope_positions = [pos for _, _, pos, _, _, _, pred in display_results['discotope'] if pred == 1]
                
                display_results['bepipred_peptides'] = extract_peptides(sequence, bp_epitope_positions, center_pos)
                display_results['discotope_peptides'] = extract_peptides(sequence, dt_epitope_positions, center_pos)
            
            tab1, tab2 = st.tabs(["BepiPred-3.0 Results", "DiscoTope-3.0 Results"])
            
            with tab1:
                if display_results['bepipred']:
                    bp_df = pd.DataFrame(display_results['bepipred'], 
                                       columns=['Position', 'Residue', 'Score', 'Prediction'])
                    epitope_df = bp_df[bp_df['Prediction'] == 'Epitope']
                    
                    subtab1, subtab2, subtab3 = st.tabs(["All Predictions", "Epitopes Only", "Peptides"])
                    
                    with subtab1:
                        st.dataframe(bp_df, use_container_width=True)
                    
                    with subtab2:
                        if not epitope_df.empty:
                            st.dataframe(epitope_df, use_container_width=True)
                            st.metric("Total Epitope Residues", len(epitope_df))
                        else:
                            st.info("No epitopes predicted with current threshold")
                    
                    with subtab3:
                        if display_results['bepipred_peptides']:
                            peptides_df = pd.DataFrame(display_results['bepipred_peptides'])
                            st.dataframe(peptides_df, use_container_width=True)
                            
                            # Summary metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Peptides", len(peptides_df))
                            with col2:
                                avg_length = peptides_df['Length'].mean() if len(peptides_df) > 0 else 0
                                st.metric("Avg Peptide Length", f"{avg_length:.1f}")
                            with col3:
                                st.metric("Center Extension", center_pos)
                        else:
                            st.info("No peptides found with current parameters")
                else:
                    st.info("BepiPred-3.0 results not available")
            
            with tab2:
                if display_results['discotope']:
                    dt_df = pd.DataFrame(display_results['discotope'], 
                                       columns=['PDB_ID', 'Chain', 'Position', 'Residue', 
                                              'Raw_Score', 'Calibrated_Score', 'Prediction'])
                    epitope_df = dt_df[dt_df['Prediction'] == 1]
                    
                    subtab1, subtab2, subtab3 = st.tabs(["All Predictions", "Epitopes Only", "Peptides"])
                    
                    with subtab1:
                        st.dataframe(dt_df, use_container_width=True)
                    
                    with subtab2:
                        if not epitope_df.empty:
                            st.dataframe(epitope_df, use_container_width=True)
                            st.metric("Total Epitope Residues", len(epitope_df))
                        else:
                            st.info("No epitopes predicted with current threshold")
                    
                    with subtab3:
                        if display_results['discotope_peptides']:
                            peptides_df = pd.DataFrame(display_results['discotope_peptides'])
                            st.dataframe(peptides_df, use_container_width=True)
                            
                            # Summary metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Peptides", len(peptides_df))
                            with col2:
                                avg_length = peptides_df['Length'].mean() if len(peptides_df) > 0 else 0
                                st.metric("Avg Peptide Length", f"{avg_length:.1f}")
                            with col3:
                                st.metric("Center Extension", center_pos)
                        else:
                            st.info("No peptides found with current parameters")
                else:
                    st.info("DiscoTope-3.0 results not available")
        
        else:
            st.error("No predictions were successful")
    
    elif run_button:
        st.warning("Please enter a UniProt ID")

if __name__ == "__main__":
    main()
