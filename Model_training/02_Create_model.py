'''
## 
## Version 6: create_model_save_parameters.py 
## Description: Late Fusion Siamese Difference Network + Logging

    # Fixed parameters obtained from optuna: Define optimized parameters as initial values within a single dictionary (allowing for easy configuration adjustments in the future).
    # CSV parameter logging: Append all hyperparameter columns to the evaluation report (results_list) to track which parameter configuration produced each specific result row.
    # Permanent saving of model & scalers: 
    # Remove the os.remove(best_model_path) command at the end to permanently keep the best performing .pt model file.

Inputs obtained from 01_extract_features_late_fusion.py
    Folder: Output_tensors
    dataset_metadata_log.csv


Outputs
    saved_optimized_model.pt: The weights of the best performing model.  
    saved_scalers.pt: Saved scaler statistics used during inference scripts to apply Z-score normalization on new data based on the training data baseline.  
    late_fusion_optimized_results.csv: Test set evaluation results with appended hyperparameter columns.


Opt2
    Merge scalers (mean and SD of train data; saved_scalers.pt) directly into the model file.

'''

import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# =====================================================================
# 1. Shared feature encoder for each branch
# =====================================================================
class SharedHybridEncoder(nn.Module):
    def __init__(self, in_features, embed_dim=128, nhead=4, num_layers=2, dropout=0.2, cnn_dropout=0.3):
        super().__init__()
        
        self.local_cnn = nn.Sequential(
            nn.Conv1d(in_features, embed_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(cnn_dropout),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU()
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.attention_weights = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: [batch, 500, in_features] -> [batch, in_features, 500]
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.local_cnn(x_cnn)
        x_trans = x_cnn.permute(0, 2, 1)
        x_trans = self.transformer(x_trans)
        
        # ดึงฟีเจอร์ตรงกลาง (ตำแหน่งที่ 250) และภาพรวมด้วย Attention
        center_idx = x_trans.size(1) // 2
        center_repr = x_trans[:, center_idx, :]
        
        attn_scores = F.softmax(self.attention_weights(x_trans), dim=1)
        global_repr = torch.sum(x_trans * attn_scores, dim=1)
        
        return torch.cat([center_repr, global_repr], dim=-1)

# =====================================================================
# 2. Main structure of the late fusion Siamese difference network
# =====================================================================
class LateFusionSiameseDifference_ddG(nn.Module):
    def __init__(self, dim_prost=2048, dim_esm=2560, dim_physbio=26, 
                 embed_prost=128, embed_esm=128, embed_phys=64,
                 dropout=0.2):
        super().__init__()
        
        self.prost_encoder = SharedHybridEncoder(dim_prost, embed_prost, nhead=4, num_layers=2, dropout=dropout)
        self.esm_encoder = SharedHybridEncoder(dim_esm, embed_esm, nhead=4, num_layers=2, dropout=dropout)
        self.physbio_encoder = SharedHybridEncoder(dim_physbio, embed_phys, nhead=2, num_layers=2, dropout=dropout)
        
        fusion_dim = (embed_prost * 2) + (embed_esm * 2) + (embed_phys * 2)
        
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1, bias=False)  # บังคับสมมาตรตรงจุดศูนย์
        )

    def _extract_diff(self, encoder, x_main, x_pair):
        return encoder(x_main) - encoder(x_pair)

    def forward(self, x_prost_main, x_prost_pair, x_esm_main, x_esm_pair, x_phys_main, x_phys_pair):
        diff_prost = self._extract_diff(self.prost_encoder, x_prost_main, x_prost_pair)
        diff_esm = self._extract_diff(self.esm_encoder, x_esm_main, x_esm_pair)
        diff_phys = self._extract_diff(self.physbio_encoder, x_phys_main, x_phys_pair)
        
        fusion_diff = torch.cat([diff_prost, diff_esm, diff_phys], dim=-1)
        out = 0.5 * (self.regressor(fusion_diff) - self.regressor(-fusion_diff))
        return out.squeeze(-1)

# =====================================================================
# 3. Late fusion dataset management class
# =====================================================================
class PairedLateFusionDataset(Dataset):
    def __init__(self, metadata_df, tensor_dir="Output_tensors", scalers=None):
        self.df = metadata_df.reset_index(drop=True)
        self.tensor_dir = tensor_dir
        self.scalers = scalers

    def __len__(self):
        return len(self.df)

    def _scale(self, tensor, key):
        if torch.isnan(tensor).any():
            tensor = torch.nan_to_num(tensor, nan=0.0)
        if self.scalers is not None and key in self.scalers:
            mean = self.scalers[key]['mean']
            std = self.scalers[key]['std']
            tensor = (tensor - mean) / (std + 1e-8)
            tensor = torch.nan_to_num(tensor, nan=0.0)
        return tensor

    def _load_dict(self, folder_name, file_name):
        file_path = os.path.join(self.tensor_dir, folder_name, file_name)
        if not os.path.exists(file_path):
            return None
        data = torch.load(file_path, map_location='cpu')
        return {
            'prost': self._scale(data['prost'].float(), 'prost'),
            'esm': self._scale(data['esm'].float(), 'esm'),
            'physbio': self._scale(data['physbio'].float(), 'physbio')
        }

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        main_folder = f"{row['Group']}_{row['Direction']}"
        main_file = row['Tensor_File']
        
        dict_main = self._load_dict(main_folder, main_file)
        
        pair_dir = 'rev' if row['Direction'] == 'fwd' else 'fwd'
        pair_file = main_file.replace(f"_{row['Direction']}.pt", f"_{pair_dir}.pt")
        pair_folder = f"{row['Group']}_{pair_dir}"
        
        dict_pair = self._load_dict(pair_folder, pair_file)
        
        if dict_pair is None and dict_main is not None:
            dict_pair = {k: -v for k, v in dict_main.items()}
            
        return (
            dict_main['prost'], dict_pair['prost'],
            dict_main['esm'], dict_pair['esm'],
            dict_main['physbio'], dict_pair['physbio'],
            torch.tensor(float(row['ddG']), dtype=torch.float32)
        )


def compute_late_fusion_scalers(train_df, tensor_dir="Output_tensors", max_samples=1000):
'''
Calculate a memory-efficient scaler using the online running mean/standard method 
and limiting the maximum number of samples
to prevent the system from being killed due to full RAM.
'''
    print(f" calculating RAM-saving scaler (maximum sampling {max_samples} )...", flush=True)
    
    # If there is too much data, use a random sample for calculation
    if len(train_df) > max_samples:
        sample_df = train_df.sample(n=max_samples, random_state=42)
    else:
        sample_df = train_df

    running_sum = {'prost': 0.0, 'esm': 0.0, 'physbio': 0.0}
    running_sq_sum = {'prost': 0.0, 'esm': 0.0, 'physbio': 0.0}
    total_rows = {'prost': 0, 'esm': 0, 'physbio': 0}
    
    for idx, row in sample_df.iterrows():
        file_path = os.path.join(tensor_dir, f"{row['Group']}_{row['Direction']}", row['Tensor_File'])
        if os.path.exists(file_path):
            data = torch.load(file_path, map_location='cpu')
            
            for k in ['prost', 'esm', 'physbio']:
                tensor_np = data[k].float().numpy()
                running_sum[k] += np.sum(tensor_np, axis=0)
                running_sq_sum[k] += np.sum(tensor_np ** 2, axis=0)
                total_rows[k] += tensor_np.shape[0]
                
            # Clear variables in each loop iteration to free up RAM for the system
            del data
            
    scalers = {}
    for k in ['prost', 'esm', 'physbio']:
        N = max(1, total_rows[k])
        mean_val = running_sum[k] / N
        
        # คำนวณ Variance: E[X^2] - (E[X])^2
        var_val = (running_sq_sum[k] / N) - (mean_val ** 2)
        std_val = np.sqrt(np.maximum(var_val, 1e-8))
        
        std_val[std_val < 1e-6] = 1.0
        scalers[k] = {
            'mean': torch.tensor(mean_val, dtype=torch.float32),
            'std': torch.tensor(std_val, dtype=torch.float32)
        }
        
    print("Scaler calculation successful (uses low RAM and is safe from being killed).)!", flush=True)
    return scalers

# =====================================================================
# 4. Traning function (parameters are saved and the model is kept)
# =====================================================================
def run_late_fusion_experiment(config, df_train, df_val, df_all, scalers, device, run_id):
    print(f"\n" + "="*60, flush=True)
    print(f"Start running the model {run_id} | Parameters: {config}", flush=True)
    print("="*60, flush=True)
    
    train_dataset = PairedLateFusionDataset(df_train, scalers=scalers)
    val_dataset = PairedLateFusionDataset(df_val, scalers=scalers)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Check the dimensions of the input data
    s_prost_m, _, s_esm_m, _, s_phys_m, _, _ = next(iter(DataLoader(train_dataset, batch_size=1)))
    dim_prost, dim_esm, dim_physbio = s_prost_m.shape[2], s_esm_m.shape[2], s_phys_m.shape[2]
    
    model = LateFusionSiameseDifference_ddG(
        dim_prost=dim_prost, dim_esm=dim_esm, dim_physbio=dim_physbio,
        embed_prost=config['embed_prost'], embed_esm=config['embed_esm'], embed_phys=config['embed_phys'],
        dropout=config['dropout']
    ).to(device)    
    
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    patience_counter = 0

    best_model_path = f"saved_{run_id}.pt"                                 #################################################################################################
    epochs_run = 0

    for epoch in range(config['epochs']):
        epochs_run = epoch + 1
        model.train()
        train_loss, valid_batches = 0.0, 0
        
        for p_m, p_p, e_m, e_p, ph_m, ph_p, targets in train_loader:
            p_m, p_p = p_m.to(device), p_p.to(device)
            e_m, e_p = e_m.to(device), e_p.to(device)
            ph_m, ph_p = ph_m.to(device), ph_p.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(p_m, p_p, e_m, e_p, ph_m, ph_p)
            loss = criterion(outputs, targets)
            
            if torch.isnan(loss): continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * targets.size(0)
            valid_batches += targets.size(0)
            
        model.eval()
        val_loss, val_samples = 0.0, 0
        with torch.no_grad():
            for p_m, p_p, e_m, e_p, ph_m, ph_p, targets in val_loader:
                p_m, p_p = p_m.to(device), p_p.to(device)
                e_m, e_p = e_m.to(device), e_p.to(device)
                ph_m, ph_p = ph_m.to(device), ph_p.to(device)
                targets = targets.to(device)
                
                loss = criterion(model(p_m, p_p, e_m, e_p, ph_m, ph_p), targets)
                if not torch.isnan(loss):
                    val_loss += loss.item() * targets.size(0)
                    val_samples += targets.size(0)
                
        train_loss /= max(1, valid_batches)
        val_loss /= max(1, val_samples)
        
        print(f"Epoch [{epochs_run:02d}/{config['epochs']}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", flush=True)
        
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            patience_counter = 0
            
            # Edit model save point: Save the model with its structural parameters
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': config,  # 
                'dim_features': {
                    'dim_prost': dim_prost,
                    'dim_esm': dim_esm,
                    'dim_physbio': dim_physbio
                },
                'scalers': scalers  # 
            }
            torch.save(checkpoint, best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early Stopping at Epoch {epochs_run}", flush=True)
                break

    # Modified model loading point: Extracted state_dict from checkpoint for testing in Testset
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    model.eval()
    # test loop in the Testset and returning the original value

    test_groups = ['Testset1', 'Testset2', 'Testset_Myoglobin', 'Testset_p53']
    results_list = []
    
    for group in test_groups:
        for direction in ['fwd', 'rev']:
            df_sub = df_all[(df_all['Group'] == group) & (df_all['Direction'] == direction)]
            if len(df_sub) == 0: continue
                
            test_loader = DataLoader(PairedLateFusionDataset(df_sub, scalers=scalers), batch_size=32, shuffle=False)
            all_preds, all_targets = [], []
            
            with torch.no_grad():
                for p_m, p_p, e_m, e_p, ph_m, ph_p, targets in test_loader:
                    preds = model(p_m.to(device), p_p.to(device), e_m.to(device), e_p.to(device), ph_m.to(device), ph_p.to(device))
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.numpy())
                    
            if len(all_targets) == 0: continue
            all_preds = np.nan_to_num(all_preds, nan=0.0)
            rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
            try:
                r_score, _ = pearsonr(all_targets, all_preds)
                if np.isnan(r_score): r_score = 0.0
            except: r_score = np.nan
                
            # Add hyperparameters to the results report
            results_list.append({
                'Run_ID': run_id,
                'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Test_Group': group, 'Direction': direction, 'Sample_Count': len(df_sub),
                'RMSE': round(rmse, 4), 'Pearson_R': round(r_score if not np.isnan(r_score) else 0.0, 4),
                'Best_Val_Loss': round(best_val_loss, 4), 'Epochs_Run': epochs_run,
                'embed_prost': config['embed_prost'],
                'embed_esm': config['embed_esm'],
                'embed_phys': config['embed_phys'],
                'lr': config['lr'],
                'dropout': config['dropout'],
                'batch_size': config['batch_size'],
                'weight_decay': config['weight_decay']
            })
            

    print(f"Save the model here: {best_model_path}", flush=True)
    return results_list


# =====================================================================
# 5. Main Function (Updated with Fixed Params and Model Saving)
# =====================================================================
def main():
    print("Starting model training", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Processing on: {device}", flush=True)
    
    output_report_file = "saved_optimized_results.csv"
    log_file = "dataset_metadata_log.csv"
    
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"The {log_file} file was not found. Please run script 01_Extract_features_late_fusion.py first")
        
    df_all = pd.read_csv(log_file)
    
    existing_groups = df_all['Group'].astype(str).str.lower().unique()
    has_val_group = any(g in ['validation', 'val'] for g in existing_groups)
    
    if has_val_group:
        print("'Validation/Val' data group was detected in the log file -> The existing data group will be used", flush=True)
        df_train = df_all[df_all['Group'].astype(str).str.lower() == 'train'].reset_index(drop=True)
        df_val = df_all[df_all['Group'].astype(str).str.lower().isin(['validation', 'val'])].reset_index(drop=True)
    else:
        print("No 'Validation' group found in the log file -> Randomly dividing from the training set (15% proportion based on PDB ID)", flush=True)
        df_train_full = df_all[df_all['Group'].astype(str).str.lower() == 'train']
        
        unique_pdbs = df_train_full['Tensor_File'].apply(lambda x: str(x).split('_')[0]).unique()
        np.random.seed(42)
        np.random.shuffle(unique_pdbs)
        
        val_size = max(1, int(len(unique_pdbs) * 0.15))
        val_pdbs = unique_pdbs[:val_size]
        
        df_val = df_train_full[df_train_full['Tensor_File'].apply(lambda x: str(x).split('_')[0]).isin(val_pdbs)].reset_index(drop=True)
        df_train = df_train_full[~df_train_full['Tensor_File'].apply(lambda x: str(x).split('_')[0]).isin(val_pdbs)].reset_index(drop=True)

    print(f"Number of data points -> Train: {len(df_train)} rows | Validation: {len(df_val)} rows", flush=True)

    # 1. Calculate the scalers and save them for future inferring
    scalers = compute_late_fusion_scalers(df_train)
    scaler_save_path = "saved_scalers.pt"                    ################################################################################################
    torch.save(scalers, scaler_save_path)
    print(f"Save the scalers for future normalization here'{scaler_save_path}'", flush=True)
    
    # 2. Define the optimal parameters ######################################################################################################################

    optimal_config = {
        'embed_prost': 128, 
        'embed_esm': 128, 
        'embed_phys': 64,
        'lr': 5.00e-05, 
        'dropout': 0.1, 
        'batch_size': 64, 
        'weight_decay': 1.00e-03, 
        'epochs': 45, 
        'patience': 10
    }
    
    # 3. Run the model training
    run_id = "optimized_model"
    run_results = run_late_fusion_experiment(optimal_config, df_train, df_val, df_all, scalers, device, run_id=run_id)
    
    # 4. Save the results to a CSV file
    pd.DataFrame(run_results).to_csv(output_report_file, index=False, encoding='utf-8-sig')
    print(f"The output report, along with its parameters, has been successfully saved to the file '{output_report_file}' ", flush=True)
    print("\nThe process is complete! You can now use 'saved_optimized_model.pt' and 'saved_scalers.pt' ", flush=True)

if __name__ == "__main__":
    main()
