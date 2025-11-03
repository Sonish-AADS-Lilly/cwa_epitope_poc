import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List
from data_retrieval import get_uniprot_sequence, get_structure_content
from bepipred_predictor import BepiPredPredictor
from discotope_predictor import DiscoTopePredictor

def extract_peptides_simple(sequence: str, epitope_positions: List[int], center_position: int = 4, min_length: int = 3) -> List[dict]:
    peptides = []
    epitope_positions = sorted(epitope_positions)
    
    if not epitope_positions:
        return peptides
    
    groups = []
    current_group = [epitope_positions[0]]
    
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
        
        peptides.append({
            'Peptide': peptide_seq,
            'Start': start_pos + 1,
            'End': end_pos,
            'Length': len(peptide_seq),
            'Core_Epitope': sequence[group[0]-1:group[-1]],
            'Positions': f"{group[0]}-{group[-1]}"
        })
    
    return peptides

st.set_page_config(page_title="Epitope Prediction Platform", layout="wide")

def main():
    st.title("Epitope Prediction Platform")
    st.markdown("BepiPred-3.0 (sequence-based) and DiscoTope-3.0 (structure-based) predictions")
    
    with st.sidebar:
        st.header("Input")
        uniprot_id = st.text_input("UniProt ID", value="P00779")
        
        st.header("Options")
        bepipred_threshold = st.slider("BepiPred-3.0 Threshold", 0.0, 1.0, 0.1512, 0.001)
        discotope_threshold = st.slider("DiscoTope-3.0 Threshold", 0.0, 2.0, 0.90, 0.01)
        
        run_prediction = st.button("Run Predictions", type="primary")
    
    if run_prediction and uniprot_id:
        run_predictions(uniprot_id, bepipred_threshold, discotope_threshold)
    elif run_prediction and not uniprot_id:
        st.error("Please enter a UniProt ID")

def run_predictions(uniprot_id: str, bepipred_threshold: float, discotope_threshold: float):
    try:
        with st.spinner("Retrieving protein data..."):
            sequence = get_uniprot_sequence(uniprot_id)
            structure_content = get_structure_content(uniprot_id, 'alphafold')
        
        st.success(f"Retrieved protein {uniprot_id} (Length: {len(sequence)} residues)")
        
        if structure_content:
            st.info("Structure type: AlphaFold prediction")
        else:
            st.warning("No AlphaFold structure available")
        
        st.header("BepiPred-3.0 (Sequence-based)")
        run_bepipred(sequence, uniprot_id, bepipred_threshold)
        
        st.header("DiscoTope-3.0 (Structure-based)")
        if structure_content:
            run_discotope(structure_content, 'alphafold', uniprot_id, discotope_threshold)
        else:
            st.warning("No structure available for this protein")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

def run_bepipred(sequence: str, uniprot_id: str, threshold: float):
    try:
        predictor = BepiPredPredictor()
        
        if not predictor.available:
            st.error("BepiPred-3.0 is not available")
            return
            
        with st.spinner("Running BepiPred-3.0..."):
            results = predictor.predict_epitopes(sequence, uniprot_id, threshold)
        
        if results:
            df = pd.DataFrame(results, columns=["Position", "Residue", "Score", "Prediction"])
            epitope_count = len(df[df["Prediction"] == "Epitope"])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Epitopes", epitope_count)
            with col2:
                st.metric("Total Residues", len(df))
            with col3:
                st.metric("Epitope Percentage", f"{epitope_count/len(df)*100:.1f}%")
            
            st.subheader("Parameter Controls")
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                new_threshold = st.number_input(
                    "Threshold:", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=threshold, 
                    step=0.001, 
                    format="%.3f",
                    key="bp_threshold_input"
                )
            
            with col2:
                center_position = st.number_input(
                    "Center position:", 
                    min_value=0, 
                    max_value=10, 
                    value=4, 
                    step=1,
                    key="bp_center_position"
                )
            
            with col3:
                if st.button("Recalculate", key="bp_recalculate"):
                    if new_threshold != threshold:
                        new_results = predictor.predict_epitopes(sequence, uniprot_id, new_threshold)
                        if new_results:
                            df = pd.DataFrame(new_results, columns=["Position", "Residue", "Score", "Prediction"])
                            epitope_count = len(df[df["Prediction"] == "Epitope"])
                            st.experimental_rerun()
            
            fig = px.bar(df, x="Position", y="Score", color="Prediction",
                        title="BepiPred-3.0 Epitope Scores",
                        color_discrete_map={"Epitope": "#ff6b6b", "Non-Epitope": "#4ecdc4"},
                        height=500)
            fig.add_hline(y=new_threshold if 'new_threshold' in locals() else threshold, 
                         line_dash="dash", 
                         annotation_text=f"Threshold ({new_threshold if 'new_threshold' in locals() else threshold:.4f})")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Results")
            tab1, tab2, tab3 = st.tabs(["Predictions", "Epitopes Only", "Peptides"])
            
            with tab1:
                st.dataframe(df, height=300, use_container_width=True)
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download BepiPred Results",
                    data=csv,
                    file_name=f"bepipred_{uniprot_id}.csv",
                    mime="text/csv"
                )
            
            with tab2:
                epitope_df = df[df["Prediction"] == "Epitope"]
                if not epitope_df.empty:
                    st.dataframe(epitope_df, height=300, use_container_width=True)
                    
                    epitope_csv = epitope_df.to_csv(index=False)
                    st.download_button(
                        label="Download Epitopes Only",
                        data=epitope_csv,
                        file_name=f"bepipred_epitopes_{uniprot_id}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No epitopes predicted with current threshold")
            
            with tab3:
                epitope_positions = df[df["Prediction"] == "Epitope"]["Position"].tolist()
                if epitope_positions:
                    peptides = extract_peptides_simple(sequence, epitope_positions, center_position)
                    if peptides:
                        peptide_df = pd.DataFrame(peptides)
                        st.dataframe(peptide_df, height=300, use_container_width=True)
                        
                        peptide_csv = peptide_df.to_csv(index=False)
                        st.download_button(
                            label="Download Peptides",
                            data=peptide_csv,
                            file_name=f"bepipred_peptides_{uniprot_id}.csv",
                            mime="text/csv"
                        )
                        
                        st.info(f"Extracted {len(peptides)} peptides with center position ±{center_position}")
                    else:
                        st.info("No peptides could be extracted from current predictions")
                else:
                    st.info("No epitopes predicted - cannot extract peptides")
                    
        else:
            st.error("BepiPred prediction failed")
    
    except Exception as e:
        st.error(f"BepiPred error: {str(e)}")

def run_discotope(structure_content: str, structure_type: str, uniprot_id: str, threshold: float):
    try:
        predictor = DiscoTopePredictor()
        
        if not predictor.available:
            st.error("DiscoTope-3.0 is not available")
            return
            
        with st.spinner("Running DiscoTope-3.0..."):
            results = predictor.predict_epitopes(structure_content, structure_type, threshold)
        
        if results:
            if hasattr(predictor, 'using_fallback') and predictor.using_fallback:
                st.warning("DiscoTope-3.0 Fallback Mode Active")
                st.info("Official DiscoTope-3.0 encountered issues. Using structure-based heuristic predictions.")
            else:
                st.success("DiscoTope-3.0 Official Model completed successfully")
            
            df = pd.DataFrame(results, columns=[
                "PDB_ID", "Chain", "Position", "Residue", 
                "Raw_Score", "Calibrated_Score", "Prediction"
            ])
            df["Epitope"] = df["Prediction"].map({1: "Epitope", 0: "Non-Epitope"})
            epitope_count = len(df[df["Prediction"] == 1])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Epitopes", epitope_count)
            with col2:
                st.metric("Total Residues", len(df))
            with col3:
                st.metric("Epitope Percentage", f"{epitope_count/len(df)*100:.1f}%")
            
            st.subheader("Parameter Controls")
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                new_threshold = st.number_input(
                    "Threshold:", 
                    min_value=0.0, 
                    max_value=2.0, 
                    value=threshold, 
                    step=0.01, 
                    format="%.2f",
                    key="dt_threshold_input"
                )
            
            with col2:
                center_position = st.number_input(
                    "Center position:", 
                    min_value=0, 
                    max_value=10, 
                    value=4, 
                    step=1,
                    key="dt_center_position"
                )
            
            with col3:
                if st.button("Recalculate", key="dt_recalculate"):
                    if new_threshold != threshold:
                        new_results = predictor.predict_epitopes(structure_content, structure_type, new_threshold)
                        if new_results:
                            df = pd.DataFrame(new_results, columns=[
                                "PDB_ID", "Chain", "Position", "Residue", 
                                "Raw_Score", "Calibrated_Score", "Prediction"
                            ])
                            df["Epitope"] = df["Prediction"].map({1: "Epitope", 0: "Non-Epitope"})
                            epitope_count = len(df[df["Prediction"] == 1])
                            st.experimental_rerun()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["Position"], 
                y=df["Calibrated_Score"],
                mode='markers',
                marker=dict(
                    color=df["Prediction"], 
                    colorscale=[[0, '#4ecdc4'], [1, '#ff6b6b']],
                    size=8
                ),
                text=df["Residue"],
                customdata=df["Chain"],
                hovertemplate="Chain %{customdata}<br>Position: %{x}<br>Score: %{y:.3f}<br>Residue: %{text}<extra></extra>"
            ))
            fig.add_hline(y=new_threshold if 'new_threshold' in locals() else threshold, 
                         line_dash="dash", 
                         annotation_text=f"Threshold ({new_threshold if 'new_threshold' in locals() else threshold:.2f})")
            fig.update_layout(
                title="DiscoTope-3.0 Epitope Scores",
                xaxis_title="Residue Position",
                yaxis_title="Calibrated Score",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Results")
            tab1, tab2, tab3 = st.tabs(["Predictions", "Epitopes Only", "Peptides"])
            
            with tab1:
                display_df = df[["Chain", "Position", "Residue", "Raw_Score", "Calibrated_Score", "Epitope"]]
                st.dataframe(display_df, height=300, use_container_width=True)
                
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="Download DiscoTope Results",
                    data=csv,
                    file_name=f"discotope_{uniprot_id}.csv",
                    mime="text/csv"
                )
            
            with tab2:
                epitope_df = df[df["Prediction"] == 1]
                if not epitope_df.empty:
                    epitope_display = epitope_df[["Chain", "Position", "Residue", "Raw_Score", "Calibrated_Score", "Epitope"]]
                    st.dataframe(epitope_display, height=300, use_container_width=True)
                    
                    epitope_csv = epitope_display.to_csv(index=False)
                    st.download_button(
                        label="Download Epitopes Only",
                        data=epitope_csv,
                        file_name=f"discotope_epitopes_{uniprot_id}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No epitopes predicted with current threshold")
            
            with tab3:
                epitope_positions = df[df["Prediction"] == 1]["Position"].tolist()
                if epitope_positions:
                    sequence_residues = df.sort_values('Position')['Residue'].tolist()
                    sequence = ''.join(sequence_residues)
                    
                    peptides = extract_peptides_simple(sequence, epitope_positions, center_position)
                    if peptides:
                        peptide_df = pd.DataFrame(peptides)
                        st.dataframe(peptide_df, height=300, use_container_width=True)
                        
                        peptide_csv = peptide_df.to_csv(index=False)
                        st.download_button(
                            label="Download Peptides",
                            data=peptide_csv,
                            file_name=f"discotope_peptides_{uniprot_id}.csv",
                            mime="text/csv"
                        )
                        
                        st.info(f"Extracted {len(peptides)} peptides with center position ±{center_position}")
                    else:
                        st.info("No peptides could be extracted from current predictions")
                else:
                    st.info("No epitopes predicted - cannot extract peptides")
                    
        else:
            st.error("DiscoTope prediction failed")
    
    except Exception as e:
        error_msg = str(e)
        if "segmentation fault" in error_msg.lower() or "ESM-IF1" in error_msg:
            st.error("DiscoTope-3.0 Failed Due to Model Issues")
            st.warning(
                "The official DiscoTope-3.0 ESM-IF1 model encountered a segmentation fault. "
                "This is a known issue with the official implementation. "
                "Possible solutions:\n"
                "- Try a different protein structure\n" 
                "- Use structures with fewer residues\n"
                "- Check system memory availability"
            )
        else:
            st.error(f"DiscoTope error: {error_msg}")

if __name__ == "__main__":
    main()
