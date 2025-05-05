#!/bin/bash
HOME_DIR="."

cd $HOME_DIR
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and setuptools first
pip install --upgrade pip setuptools

# Install required dependencies
pip install torch torchvision torchaudio
pip install tqdm pandas matplotlib seaborn
pip install scikit-learn
pip install qiskit==1.4.2
pip install qiskit-aer
pip install qiskit-machine-learning
pip install nvidia-pyindex
pip install cuquantum-cu12

# Install the project in editable mode
pip install -e .

# Export environment variables
export PORTO_OUT_DIRECTORY="./trajectory_code/data/porto"
export PORTO_DATA_FILE="./pkdd-15-predict-taxi-service-trajectory-i/train.csv"