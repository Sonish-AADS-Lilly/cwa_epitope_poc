# Model Setup Instructions

This repository contains the source code for BepiPred-3.0 and DiscoTope-3.0, but excludes the large binary model files to keep the repository size manageable.

## Required Model Files

### BepiPred-3.0
The following directories need to be downloaded and extracted:
- `BepiPred-3.0/pip/` - Contains the BP3 Python package
- `BepiPred-3.0/misc/` - Contains additional model components

### DiscoTope-3.0
The following files need to be downloaded:
- `DiscoTope-3.0/models/` - Contains 100 XGBoost models and GAM calibration files
- `DiscoTope-3.0/models.zip` - Compressed model archive

## Setup Process

1. **Download the complete packages** from their original sources:
   - BepiPred-3.0: [Original Repository/Paper]
   - DiscoTope-3.0: [Original Repository/Paper]

2. **Extract model files** to their respective directories in this repository

3. **Install dependencies**:
   ```bash
   bash setup_demo.sh
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## File Sizes
- BepiPred-3.0 models: ~78MB
- DiscoTope-3.0 models: ~54MB
- Total additional download: ~132MB

The source code and demo infrastructure are included in this repository for convenience and integration purposes.
