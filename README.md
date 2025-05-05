# Trajectory Anomaly Detection with Quantum-Enhanced Attention

This code was adapted from the paper "Trajectory Anomaly Detection with Language Models" by [Mbuya et al.](https://arxiv.org/abs/2409.15366v1). The code implements the LMTAD model and an altered version using quantum-enhanced attention for trajectory anomaly detection.

### Python environment
The ```create_venv.sh``` has the instruction on how to create an environment to run the experiments in this repository. 

```
source create_venv.sh
```

### Data Preprocessing

#### Porto Dataset
Download the dataset from https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data

Unzip data and find the ```train.csv```. Set the variable ```$PORTO_DATA_FILE``` to the path of the ```train.csv``` that contains the porto data

```
PORTO_DATA_FILE="./pkdd-15-predict-taxi-service-trajectory-i/train.csv"
```

Then set the output directory for the preprocessed data. The output directory will contain the preprocessed data in the format required by the LMTAD model.

```
PORTO_OUT_DIRECTORY="./trajectory_code/data/porto"
```

Then run the following command to preprocess the Porto dataset. This will create a directory with the preprocessed data in the format required by the LMTAD model.

```
python -m trajectory_code.preprocess.preprocess_porto --data_dir "${PORTO_DATA_FILE}" --out_dir "${PORTO_OUT_DIRECTORY}"
```

### Training

#### LM-TAD
To train the LMTAD model, run the following command from the QuantumAttention directory
```
bash scripts/LMTAD/train.sh
```
To train the quantum model, run the following command from the root directory
```
bash scripts/LMTAD/train_quantum.sh
```

### EVALUATION

#### LM-TAD
To evaluate the LMTAD model, run the following command from the root directory
```
bash scripts/LMTAD/eval_porto.sh
```
To evaluate the quantum model, run the following command from the root directory
```
bash scripts/LMTAD/eval_porto_quantum.sh
```

### SINGLE COMMAND

To run the entire pipeline, you can use the following command from the QuantumAttention directory. This will preprocess the data, train the model, and evaluate the model in one go. 

```
(source venv/bin/activate || source create_venv.sh) &&
PORTO_OUT_DIRECTORY="./trajectory_code/data/porto" &&
PORTO_DATA_FILE="./pkdd-15-predict-taxi-service-trajectory-i/train.csv" &&
python -m trajectory_code.preprocess.preprocess_porto --data_dir "${PORTO_DATA_FILE}" --out_dir "${PORTO_OUT_DIRECTORY}" &&
bash scripts/LMTAD/train.sh &&
bash scripts/LMTAD/eval_porto.sh &&
bash scripts/LMTAD/train_quantum.sh &&
bash scripts/LMTAD/eval_porto_quantum.sh
```