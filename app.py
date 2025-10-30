import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_retrieval import get_uniprot_sequence, get_structure_content
from bepipred_predictor import BepiPredPredictor
from discotope_predictor import DiscoTopePredictor

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
            
            fig = px.bar(df, x="Position", y="Score", color="Prediction",
                        title="BepiPred-3.0 Epitope Scores",
                        color_discrete_map={"Epitope": "#ff6b6b", "Non-Epitope": "#4ecdc4"},
                        height=500)
            fig.add_hline(y=threshold, line_dash="dash", 
                         annotation_text=f"Threshold ({threshold:.4f})")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Detailed Results")
            st.dataframe(df, height=300, use_container_width=True)
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download BepiPred Results",
                data=csv,
                file_name=f"bepipred_{uniprot_id}.csv",
                mime="text/csv"
            )
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
            # Check if fallback was used and display appropriate message
            if hasattr(predictor, 'using_fallback') and predictor.using_fallback:
                st.warning("🔄 **DiscoTope-3.0 Fallback Mode Active**")
                st.info("Official DiscoTope-3.0 encountered issues. Using structure-based heuristic predictions.")
            else:
                st.success("✅ **DiscoTope-3.0 Official Model** completed successfully")
            
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
            fig.add_hline(y=threshold, line_dash="dash", 
                         annotation_text=f"Threshold ({threshold:.2f})")
            fig.update_layout(
                title="DiscoTope-3.0 Epitope Scores",
                xaxis_title="Residue Position",
                yaxis_title="Calibrated Score",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Detailed Results")
            display_df = df[["Chain", "Position", "Residue", "Raw_Score", "Calibrated_Score", "Epitope"]]
            st.dataframe(display_df, height=300, use_container_width=True)
            
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download DiscoTope Results",
                data=csv,
                file_name=f"discotope_{uniprot_id}.csv",
                mime="text/csv"
            )
        else:
            st.error("DiscoTope prediction failed")
    
    except Exception as e:
        error_msg = str(e)
        if "segmentation fault" in error_msg.lower() or "ESM-IF1" in error_msg:
            st.error("⚠️ **DiscoTope-3.0 Failed Due to Model Issues**")
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
