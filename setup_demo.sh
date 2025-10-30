#!/bin/bash

echo "Setting up Epitope Prediction Demo..."

echo "Installing basic requirements..."
pyenv activate chatwithantigen-env 
pip install -r requirements_demo.txt

echo "Installing BepiPred-3.0 dependencies..."
if [ -d "BepiPred-3.0/pip/Version12_7/BP3" ]; then
    cd BepiPred-3.0/pip/Version12_7/BP3
    pip install .
    cd ../../../../
    echo "BepiPred-3.0 installed successfully"
else
    echo "Warning: BepiPred-3.0 package not found"
fi

echo "Installing DiscoTope-3.0 dependencies..."
if [ -f "DiscoTope-3.0/requirements.txt" ]; then
    pip install -r DiscoTope-3.0/requirements.txt
    cd DiscoTope-3.0
    pip install .
    cd ..
    echo "DiscoTope-3.0 dependencies installed"
else
    echo "Warning: DiscoTope-3.0 requirements not found"
fi

echo "Checking models..."
if [ ! -f "DiscoTope-3.0/models/XGB_1_of_100.json" ]; then
    echo "Extracting DiscoTope-3.0 models..."
    cd DiscoTope-3.0
    unzip -q models.zip 2>/dev/null || echo "Note: models.zip not found or already extracted"
    cd ..
fi

echo "Setup complete!"
echo "Run the demo with: streamlit run app.py"
