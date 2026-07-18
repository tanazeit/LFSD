# LFSD
a late-fusion Siamese network integrating structural, evolutionary and biophysical modalities for predicting protein stability changes (ΔΔG)

This repository provides an automated pipeline to extract structural/sequence features and predict free energy changes (ddG) upon mutation using deep learning models (ProstT5, ESM, and Biopython). It supports both execution via CPU and GPU acceleration (CUDA).

Before running the pipeline, ensure the following tools are installed on your system. These cannot be fully resolved by Conda alone:
 - MSMS (Microscopic Surface Method): The feature extraction tool uses Biopython's ResidueDepth module, which requires the msms executable to be installed and available in your system's PATH variable.


### 🛠️ Installation & setup
Choose the environment configuration that matches your hardware capabilities.

__Create the environment__

__For NVIDIA GPU (CUDA) acceleration__
  ```bash
      conda env create -f environment_GPU.yaml
  ```
__For CPU-only execution__
  ```bash
      conda env create -f environment_CPU.yaml
  ```

__Activate the environment__
  ```bash
      conda activate predictDDG
  ```

__Structure preprocessing guide__
It is highly recommended to prepare and clean your PDB files before training or running predictions.

* Residue numbering: Ensure mutation positions match the specific Residue ID (auth_seq_id) inside the PDB file, not the zero-indexed positional array index.

* Isolate chains: Keep only the relevant target protein chain, remove heteroatoms (HETATM), and clean up formatting artifacts using pdb-tools:

```bash
    pip install pdb-tools
    pdb_selchain -A 1SHF.pdb | pdb_delhetatm | pdb_tidy > Input_structures/1shf_A.pdb
```

### Inference mode (Predicting new data)
To predict mutational effects using a pre-trained model, download the contents of the Prediction folder.

__Required directory structure__
Ensure your working folder contain all these files/folder before executing the script:
* 03_predict_ddG.py
* Input_data_for_prediction.csv 
  * The input CSV file must contain a comma-separated header block matching this schema:
  * ```bash
      PDB_file,Chain,WT,Position,Mutation
      1bz6.pdb,A,L,29,N
      1bz6.pdb,A,V,68,N
      1bz6.pdb,A,F,123,T
  ```
* Folder 'Input_structures' : PDB files must be in this folder name 

__Execution__
```bash
# 1. Using default parameters (looks for 'saved_optimized_model.pt' and 'Input_data_for_prediction.csv')
python 03_predict_ddG.py

# 2. Specifying custom model paths and datasets explicitly
python 03_predict_ddG.py --model path/to/model.pt --input test_predict.csv
```
__Expected output__
The script generates a new results file prefixed with Predictions_:
Example output name: Predictions_test_predict.csv

### Model training mode
To train the predictive network architecture with your own data from scratch using optimized hyperparameter spaces, download the Model_training folder structure.

__Required directory structure__
Ensure your working folder contain all these files/folder before executing the script:
* 01_Extract_features_late_fusion_for_predict.py
* Data_for_train_model.csv  <-- (Must use this exact file name)
  * The input CSV file must contain a comma-separated header block matching this schema:
  * ```bash
      PDB_file,Chain,WT,Position,Mutation,ddG,Group
      12ca.pdb,A,G,145,R,-0.36,Train
      1a23.pdb,A,C,30,S,-1.80,Train
      1a43.pdb,A,W,184,A,-0.70,Test
  ```
    * ddG: Experimental value (ddG).
    * Group: Partition tag (Train, Validation, or Test).  
* Folder [Input_structures] : PDB files must be in this folder name
* 02_Create_model_save_parameters.py

__Expected output__
```bash
    python 01_Extract_features_late_fusion_for_predict.py
```
Generated outputs
Upon completing feature engineering and baseline configuration steps, the script saves its state within these directories:

  * dataset_metadata_log.csv (Execution and indexing trace summaries)
  * Log_errors_train.txt (Error log generated if PDB structures fail extraction steps)
  * Folder: ProstT5_intermediate/ (Cached embedding arrays from the ProstT5 language encoder)
  * Folder: ESM_intermediate/ (Cached transformer representations from the ESM model)
  * Folder: Output_tensors/ (Finalized multi-modal feature tensors mapped for network layers)

__Train model with your data__
```bash
    python 02_Create_model.py
```

Generated outputs
  * saved_optimized_model.pt
  * saved_scalers.pt
  * dataset_metadata_log.csv
  * saved_optimized_results.csv
