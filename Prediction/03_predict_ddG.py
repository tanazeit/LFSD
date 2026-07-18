'''
##03_predict_ddG_revise1.py

Description: End-to-End Inference Script for Late Fusion Siamese Difference Network.
- Fully synchronized with 01_Extract_features_late_fusion.py and 02_Create_model.py.
- Automatically extracts TRUE reverse mutation features (Mut -> WT) on-the-fly to serve as the 
  Siamese pair input, matching the exact training and testset evaluation architecture.
- Outputs predictions strictly for the forward mutations specified in the input CSV.

Command Line Usage:
    python 03_predict_ddG.py --model saved_optimized_model.pt --input Input_data_for_prediction.csv
'''

import os
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
import multiprocessing as mp
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, EsmModel
from Bio.PDB import PDBParser, ResidueDepth
from Bio import BiopythonWarning

warnings.simplefilter('ignore', BiopythonWarning)

AA_3TO1_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

# =====================================================================
# 1. Model architecture (Must match 02_Create_model.py exactly)
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
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 2,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_weights = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.Tanh(), nn.Linear(32, 1)
        )

    def forward(self, x):
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.local_cnn(x_cnn)
        x_trans = x_cnn.permute(0, 2, 1)
        x_trans = self.transformer(x_trans)
        center_idx = x_trans.size(1) // 2
        center_repr = x_trans[:, center_idx, :]
        attn_scores = F.softmax(self.attention_weights(x_trans), dim=1)
        global_repr = torch.sum(x_trans * attn_scores, dim=1)
        return torch.cat([center_repr, global_repr], dim=-1)

class LateFusionSiameseDifference_ddG(nn.Module):
    def __init__(self, dim_prost=2048, dim_esm=2560, dim_physbio=26, 
                 embed_prost=128, embed_esm=128, embed_phys=64, dropout=0.2):
        super().__init__()
        self.prost_encoder = SharedHybridEncoder(dim_prost, embed_prost, nhead=4, num_layers=2, dropout=dropout)
        self.esm_encoder = SharedHybridEncoder(dim_esm, embed_esm, nhead=4, num_layers=2, dropout=dropout)
        self.physbio_encoder = SharedHybridEncoder(dim_physbio, embed_phys, nhead=2, num_layers=2, dropout=dropout)
        fusion_dim = (embed_prost * 2) + (embed_esm * 2) + (embed_phys * 2)
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1, bias=False)
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
# 2. Worker & Feature extraction pipeline
# =====================================================================
def worker_depth(pdb_path, chain_id, result_dict):
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("tmp", pdb_path)
        model = structure[0]
        chain = model[chain_id]
        residues = [res for res in chain if res.get_id()[0] == ' ']
        rd = ResidueDepth(model)
        depths = []
        for res in residues:
            key = (chain.get_id(), res.get_id())
            d = rd[key] if key in rd else (0.0, 0.0)
            depths.append([d[0], d[1]])
        result_dict['depths'] = depths
        result_dict['residues'] = residues
        result_dict['sequence'] = "".join([AA_3TO1_MAP.get(r.get_resname(), 'X') for r in residues])
        result_dict['success'] = True
    except Exception as e:
        result_dict['error'] = str(e)

class EndToEndInferencePipeline:
    def __init__(self, model_path, prost_model="Rostlab/ProstT5", esm_model="facebook/esm2_t33_650M_UR50D", device="cpu"):
        self.device = device
        self.prost_model_name = prost_model
        self.esm_model_name = esm_model
        self.prost_tokenizer, self.prost_model = None, None
        self.esm_tokenizer, self.esm_model = None, None
        
        self.struct_dir = "Input_structures"
        self.prost_cache_dir = "ProstT5_intermediate"
        self.esm_cache_dir = "ESM_intermediate"
        self.log_file = "Log_errors_inference.txt"
        
        os.makedirs(self.prost_cache_dir, exist_ok=True)
        os.makedirs(self.esm_cache_dir, exist_ok=True)
        
        self._load_grantham_matrix()
        self._load_protscale_features()
        
        # 1. โหลดโมเดลและเช็ค Scalers
        self.model, self.scalers = self._load_model_and_scalers(model_path)

    def _load_grantham_matrix(self):
        if os.path.exists("Score_matrix_Grantham.txt"):
            self.grantham_df = pd.read_csv("Score_matrix_Grantham.txt", sep='\t', index_col=0)
        else:
            raise FileNotFoundError("File not found: Score_matrix_Grantham.txt")

    def _load_protscale_features(self):
        if os.path.exists("Protscale_features.txt"):
            self.protscale_df = pd.read_csv("Protscale_features.txt", sep='\t', index_col=0)
            self.protscale_df.columns = [AA_3TO1_MAP.get(col.upper(), col) for col in self.protscale_df.columns]
        else:
            raise FileNotFoundError("File not found: Protscale_features.txt")

    def _load_model_and_scalers(self, model_path):
        print(f"Loading checkpoint from '{model_path}'...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'scalers' in checkpoint and checkpoint['scalers'] is not None:
            print("Found embedded scalers inside the model checkpoint! No external scaler file needed.")
            scalers = checkpoint['scalers']
        else:
            print("Scalers NOT found inside the model! Attempting to load from 'saved_scalers.pt'...")
            if os.path.exists("saved_scalers.pt"):
                scalers = torch.load("saved_scalers.pt", map_location='cpu')
                print("Successfully loaded fallback scalers from 'saved_scalers.pt'")
            else:
                raise FileNotFoundError("Cannot find embedded scalers OR 'saved_scalers.pt'! Normalization is impossible.")

        config = checkpoint.get('config', {'embed_prost': 128, 'embed_esm': 128, 'embed_phys': 64, 'dropout': 0.1})
        dims = checkpoint.get('dim_features', {'dim_prost': 2048, 'dim_esm': 2560, 'dim_physbio': 26})
        
        model = LateFusionSiameseDifference_ddG(
            dim_prost=dims['dim_prost'], dim_esm=dims['dim_esm'], dim_physbio=dims['dim_physbio'],
            embed_prost=config['embed_prost'], embed_esm=config['embed_esm'], embed_phys=config['embed_phys'],
            dropout=config['dropout']
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model architecture and weights loaded successfully!")
        return model, scalers

    def _scale_tensor(self, tensor, key):
        if torch.isnan(tensor).any():
            tensor = torch.nan_to_num(tensor, nan=0.0)
        if self.scalers is not None and key in self.scalers:
            mean = self.scalers[key]['mean'].to(tensor.device)
            std = self.scalers[key]['std'].to(tensor.device)
            tensor = (tensor - mean) / (std + 1e-8)
            tensor = torch.nan_to_num(tensor, nan=0.0)
        return tensor

    def standardize_aa(self, aa_str):
        aa_str = str(aa_str).strip().upper()
        if len(aa_str) == 3: return AA_3TO1_MAP.get(aa_str, None)
        return aa_str if len(aa_str) == 1 else None

    def get_prost5_embedding(self, sequence, cache_key):
        cache_path = os.path.join(self.prost_cache_dir, f"{cache_key}.npy")
        if os.path.exists(cache_path): return np.load(cache_path)
        if self.prost_model is None:
            print(f"Loading ProstT5: {self.prost_model_name} onto {self.device}...")
            self.prost_tokenizer = T5Tokenizer.from_pretrained(self.prost_model_name, do_lower_case=False)
            self.prost_model = T5EncoderModel.from_pretrained(self.prost_model_name).to(self.device)
            self.prost_model.eval()
        processed_seq = "<AA2fold> " + " ".join(list(sequence))
        inputs = self.prost_tokenizer(processed_seq, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.prost_model(**inputs)
        embedding = outputs.last_hidden_state[0, 1:len(sequence)+1, :].cpu().numpy()
        np.save(cache_path, embedding)
        return embedding

    def get_esm_embedding(self, sequence, cache_key):
        cache_path = os.path.join(self.esm_cache_dir, f"{cache_key}.npy")
        if os.path.exists(cache_path): return np.load(cache_path)
        if self.esm_model is None:
            print(f"Loading ESM: {self.esm_model_name} onto {self.device}...")
            self.esm_tokenizer = AutoTokenizer.from_pretrained(self.esm_model_name)
            self.esm_model = EsmModel.from_pretrained(self.esm_model_name).to(self.device)
            self.esm_model.eval()
        inputs = self.esm_tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(self.device)
        with torch.no_grad():
            outputs = self.esm_model(**inputs)
        embedding = outputs.last_hidden_state[0, 1:len(sequence)+1, :].cpu().numpy()
        np.save(cache_path, embedding)
        return embedding

    def log_error(self, message):
        print(message)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")

    def get_fixed_length_500(self, matrix, mut_idx, target_len=500):
        seq_len, dim = matrix.shape
        if seq_len <= target_len:
            padded = np.zeros((target_len, dim), dtype=np.float32)
            padded[:seq_len, :] = matrix
            return padded
        else:
            half_len = target_len // 2
            start_idx = mut_idx - half_len
            end_idx = start_idx + target_len
            if start_idx < 0:
                start_idx, end_idx = 0, target_len
            elif end_idx > seq_len:
                end_idx, start_idx = seq_len, seq_len - target_len
            return matrix[start_idx:end_idx, :]

    def _extract_branch_features(self, sequence, mut_idx, wt_aa, mu_aa, pdb_file, chain, pos, f_geom, dir_label):
        """
        Helper method to extract (ProstT5, ESM2, PhysBio) tensors for any directed mutation (fwd/rev)
        ensuring 100% consistency with script 01_Extract_features_late_fusion.py.
        """
        seq_len = len(sequence)
        seq_base = sequence[:mut_idx] + wt_aa + sequence[mut_idx+1:]
        seq_mut = sequence[:mut_idx] + mu_aa + sequence[mut_idx+1:]
        
        # 1. ProstT5 Branch
        f_prost_base = self.get_prost5_embedding(seq_base, f"{pdb_file}_{chain}_{wt_aa}{pos}_PROST")[:seq_len, :]
        f_prost_mut = self.get_prost5_embedding(seq_mut, f"{pdb_file}_{chain}_{mu_aa}{pos}_PROST")[:seq_len, :]
        feat_prost = np.hstack([f_prost_base, f_prost_mut - f_prost_base])

        # 2. ESM2 Branch
        f_esm_base = self.get_esm_embedding(seq_base, f"{pdb_file}_{chain}_{wt_aa}{pos}_ESM")[:seq_len, :]
        f_esm_mut = self.get_esm_embedding(seq_mut, f"{pdb_file}_{chain}_{mu_aa}{pos}_ESM")[:seq_len, :]
        feat_esm = np.hstack([f_esm_base, f_esm_mut - f_esm_base])
        
        # 3. Biophysical & Structural Branch
        f_protscale_base = np.array([self.protscale_df.get(aa, pd.Series(np.zeros(len(self.protscale_df)))).values for aa in seq_base], dtype=np.float32)
        try:
            grantham_dist = self.grantham_df.loc[wt_aa, mu_aa]
        except KeyError:
            grantham_dist = 0.0
        protscale_wt_val = self.protscale_df.get(wt_aa, pd.Series(np.zeros(len(self.protscale_df)))).values
        protscale_mu_val = self.protscale_df.get(mu_aa, pd.Series(np.zeros(len(self.protscale_df)))).values
        protscale_sub = protscale_mu_val - protscale_wt_val
        
        f_mutation_profile = np.concatenate([[grantham_dist], protscale_wt_val, protscale_mu_val, protscale_sub])
        f_mut_matrix = np.tile(f_mutation_profile, (seq_len, 1)).astype(np.float32)
        feat_physbio = np.hstack([f_protscale_base, f_geom, f_mut_matrix])

        # Apply padding / cropping to fixed length 500
        t_prost = torch.tensor(self.get_fixed_length_500(feat_prost, mut_idx, target_len=500), dtype=torch.float32)
        t_esm = torch.tensor(self.get_fixed_length_500(feat_esm, mut_idx, target_len=500), dtype=torch.float32)
        t_physbio = torch.tensor(self.get_fixed_length_500(feat_physbio, mut_idx, target_len=500), dtype=torch.float32)
        
        return t_prost, t_esm, t_physbio

    def predict_from_csv(self, input_csv):
        df = pd.read_csv(input_csv)
        print(f"\nStarting extraction and prediction for {len(df)} rows from '{input_csv}'...")
        
        predictions = []

        for idx, row in df.iterrows():
            try:
                pdb_file = row['PDB_file'] if str(row['PDB_file']).endswith('.pdb') else str(row['PDB_file']) + ".pdb"
                chain, pos = str(row['Chain']).strip(), int(row['Position'])
                wt, mu = self.standardize_aa(row['WT']), self.standardize_aa(row['Mutation'])

                manager = mp.Manager()
                res_dict = manager.dict({'success': False, 'depths': [], 'residues': [], 'sequence': '', 'error': ''})
                p = mp.Process(target=worker_depth, args=(os.path.join(self.struct_dir, pdb_file), chain, res_dict))
                p.start(); p.join(timeout=10)
                
                if p.is_alive(): 
                    p.terminate()
                    self.log_error(f"Skipping index [{idx+1}]: process freezes {pdb_file}")
                    predictions.append(np.nan); continue
                if not res_dict['success']: 
                    self.log_error(f"Skipping index [{idx+1}]: failed to retrieve {pdb_file} ({res_dict['error']})")
                    predictions.append(np.nan); continue

                mut_idx = next((i for i, r in enumerate(res_dict['residues']) if r.get_id()[1] == pos), -1)
                if mut_idx == -1: 
                    self.log_error(f"Skipping index [{idx+1}]: {pdb_file} not found residue at position {pos}")
                    predictions.append(np.nan); continue
                if res_dict['sequence'][mut_idx] != wt: 
                    self.log_error(f"Skipping index [{idx+1}]: {pdb_file} mismatch (found {res_dict['sequence'][mut_idx]} but expected {wt})")
                    predictions.append(np.nan); continue

                sequence = res_dict['sequence']
                seq_len = len(sequence)

                f_geom = np.array(res_dict['depths'], dtype=np.float32)[:seq_len, :]
                if len(f_geom) < seq_len:
                    f_geom = np.vstack([f_geom, np.zeros((seq_len - len(f_geom), 2), dtype=np.float32)])

                # ----------------------------------------------------------------------------------
                # 1. Extract Forward features (WT -> Mut) to act as MAIN input
                # ----------------------------------------------------------------------------------
                t_prost_main, t_esm_main, t_phys_main = self._extract_branch_features(
                    sequence, mut_idx, wt, mu, pdb_file, chain, pos, f_geom, dir_label="fwd"
                )
                
                # ----------------------------------------------------------------------------------
                # 2. Extract Reverse features (Mut -> WT) to act as PAIR input
                # ----------------------------------------------------------------------------------
                t_prost_pair, t_esm_pair, t_phys_pair = self._extract_branch_features(
                    sequence, mut_idx, mu, wt, pdb_file, chain, pos, f_geom, dir_label="rev"
                )

                # Process Tensors & Normalization (Main & Pair)
                p_m = self._scale_tensor(t_prost_main, 'prost').unsqueeze(0).to(self.device)
                p_p = self._scale_tensor(t_prost_pair, 'prost').unsqueeze(0).to(self.device)
                
                e_m = self._scale_tensor(t_esm_main, 'esm').unsqueeze(0).to(self.device)
                e_p = self._scale_tensor(t_esm_pair, 'esm').unsqueeze(0).to(self.device)
                
                ph_m = self._scale_tensor(t_phys_main, 'physbio').unsqueeze(0).to(self.device)
                ph_p = self._scale_tensor(t_phys_pair, 'physbio').unsqueeze(0).to(self.device)

                # Predict ddG using true Siamese pairs (returning only the forward prediction!)
                with torch.no_grad():
                    pred_ddg = self.model(p_m, p_p, e_m, e_p, ph_m, ph_p).item()

                predictions.append(pred_ddg)
                print(f"Predicted [{idx+1}/{len(df)}] {pdb_file} {wt}{pos}{mu} -> ddG: {pred_ddg:.4f}")

            except Exception as e:
                self.log_error(f"Skipping [{idx+1}]: error occurred -> {e}")
                predictions.append(np.nan)

        # Save Results (Only forward predictions corresponding to the input CSV)
        df['Predicted_ddG'] = predictions
        output_csv = f"Predictions_{os.path.basename(input_csv)}"
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\nDone! All predictions have been saved to '{output_csv}'.")

# =====================================================================
# 3. Main function with argument parsing
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Late Fusion Siamese Network Prediction Script")
    parser.add_argument("--model", type=str, default="saved_optimized_model.pt", help="Path to the saved .pt model file")
    parser.add_argument("--input", type=str, default="Input_data_for_prediction.csv", help="Path to the input CSV file for prediction")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference on device: {device}")
    
    if not os.path.exists(args.input):
        print(f"Input CSV file '{args.input}' not found. Please provide a valid CSV file.")
    else:
        pipeline = EndToEndInferencePipeline(model_path=args.model, device=device)
        pipeline.predict_from_csv(args.input)