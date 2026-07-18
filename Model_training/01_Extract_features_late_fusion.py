'''
Description: Features were categorized into three distinct groups (ProstT5, ESM2, and PhysBio) 
to support a Late Fusion / Ensemble Model framework. To ensure uniform input dimensions, 
a centered padding and cropping mechanism was implemented, yielding a fixed matrix length of 500 amino acids 
with the mutation site positioned symmetrically at the center.

'''

import os
import datetime
import numpy as np
import pandas as pd
import torch
import warnings
import multiprocessing as mp
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, EsmModel
from Bio.PDB import PDBParser, ResidueDepth
from Bio import BiopythonWarning

warnings.simplefilter('ignore', BiopythonWarning)

metadata_list = []

AA_3TO1_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

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

class GranthamProtscaleLateFusionPipeline:
    def __init__(self, prost_model="Rostlab/ProstT5", esm_model="facebook/esm2_t33_650M_UR50D", device="cpu"):
        self.device = device
        
        # ProstT5 
        self.prost_model_name = prost_model
        self.prost_tokenizer = None
        self.prost_model = None
        
        # ESM
        self.esm_model_name = esm_model
        self.esm_tokenizer = None
        self.esm_model = None
        
        # Directories
        self.struct_dir = "Input_structures"
        self.prost_cache_dir = "ProstT5_intermediate"
        self.esm_cache_dir = "ESM_intermediate"
        self.output_dir = "Output_tensors"  # 
        self.log_file = "Log_errors_v0.txt"
        
        os.makedirs(self.prost_cache_dir, exist_ok=True)
        os.makedirs(self.esm_cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self._load_grantham_matrix()
        self._load_protscale_features()
        
        self.reported_shapes = False

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
        """
        Adjust the Matrix length to a fixed 500
        - If shorter than 500: add Zero-padding 
        - If longer than 500: Trim both ends so that the mutation point (mut_idx) is centered.
        """
        seq_len, dim = matrix.shape
        
        if seq_len <= target_len:
            padded = np.zeros((target_len, dim), dtype=np.float32)
            padded[:seq_len, :] = matrix
            return padded
        else:
            half_len = target_len // 2  # 250
            start_idx = mut_idx - half_len
            end_idx = start_idx + target_len
            
            # in cases where the mutation site is too close to the N-terminal
            if start_idx < 0:
                start_idx = 0
                end_idx = target_len
            # in cases where the mutation site is too close to the C-terminal
            elif end_idx > seq_len:
                end_idx = seq_len
                start_idx = seq_len - target_len
                
            return matrix[start_idx:end_idx, :]

    def print_feature_report(self, feat_prost, feat_esm, feat_physbio):
        print("\n" + "="*70)
        print("LATE FUSION FEATURE DIMENSION REPORT")
        print("="*70)
        print(f"1. ProstT5 Branch (Base+Diff) shape : {feat_prost.shape} -> ({feat_prost.shape[1]} features)")
        print(f"2. ESM2 Branch (Base+Diff) shape    : {feat_esm.shape} -> ({feat_esm.shape[1]} features)")
        print(f"3. PhysBio Branch shape             : {feat_physbio.shape} -> ({feat_physbio.shape[1]} features)")
        print("-" * 70)
        print(f"🔹 Fixed Sequence Length            : 500 AA (Zero-padded or Centered-cropped)")
        print(f"🔹 Output Tensor Structure          : Dictionary containing 3 distinct tensors")
        print("="*70 + "\n")

    def process_and_save_all(self, input_csv):
        df = pd.read_csv(input_csv)
        print(f"Starting to process {len(df)} rows of data for late fusion model...")

        for idx, row in df.iterrows():
            try:
                pdb_file = row['PDB_file'] if row['PDB_file'].endswith('.pdb') else row['PDB_file'] + ".pdb"
                chain, pos = str(row['Chain']).strip(), int(row['Position'])
                wt, mu, group = self.standardize_aa(row['WT']), self.standardize_aa(row['Mutation']), row['Group']
                ddg = float(row['ddG'])

                manager = mp.Manager()
                res_dict = manager.dict({'success': False, 'depths': [], 'residues': [], 'sequence': '', 'error': ''})
                p = mp.Process(target=worker_depth, args=(os.path.join(self.struct_dir, pdb_file), chain, res_dict))
                p.start(); p.join(timeout=10)
                
                if p.is_alive(): 
                    p.terminate()
                    self.log_error(f"Skipping index [{idx+1}]: process freezes {pdb_file}")
                    continue
                if not res_dict['success']: 
                    self.log_error(f"Skipping index [{idx+1}]: failed to retrieved {pdb_file} for ({res_dict['error']})")
                    continue

                mut_idx = next((i for i, r in enumerate(res_dict['residues']) if r.get_id()[1] == pos), -1)
                if mut_idx == -1: 
                    self.log_error(f"Skipping index [{idx+1}]: {pdb_file} not found residue at position {pos}")
                    continue
                if res_dict['sequence'][mut_idx] != wt: 
                    self.log_error(f"Skipping index [{idx+1}]: {pdb_file} original amino acid mismatch (found {res_dict['sequence'][mut_idx]} but {wt} was specified)")
                    continue

                sequence = res_dict['sequence']
                seq_len = len(sequence)

                f_geom = np.array(res_dict['depths'], dtype=np.float32)[:seq_len, :]
                if len(f_geom) < seq_len:
                    f_geom = np.vstack([f_geom, np.zeros((seq_len - len(f_geom), 2), dtype=np.float32)])

                directions = [
                    {'dir_name': 'fwd', 'wt_aa': wt, 'mu_aa': mu, 'ddg': ddg},
                    {'dir_name': 'rev', 'wt_aa': mu, 'mu_aa': wt, 'ddg': -ddg}
                ]

                for d in directions:
                    seq_base = sequence[:mut_idx] + d['wt_aa'] + sequence[mut_idx+1:]
                    seq_mut = sequence[:mut_idx] + d['mu_aa'] + sequence[mut_idx+1:]
                    
                    # 1. ProstT5 Branch
                    f_prost_base = self.get_prost5_embedding(seq_base, f"{pdb_file}_{chain}_{d['wt_aa']}{pos}_PROST")[:seq_len, :]
                    f_prost_mut = self.get_prost5_embedding(seq_mut, f"{pdb_file}_{chain}_{d['mu_aa']}{pos}_PROST")[:seq_len, :]
                    feat_prost = np.hstack([f_prost_base, f_prost_mut - f_prost_base])

                    # 2. ESM2 Branch
                    f_esm_base = self.get_esm_embedding(seq_base, f"{pdb_file}_{chain}_{d['wt_aa']}{pos}_ESM")[:seq_len, :]
                    f_esm_mut = self.get_esm_embedding(seq_mut, f"{pdb_file}_{chain}_{d['mu_aa']}{pos}_ESM")[:seq_len, :]
                    feat_esm = np.hstack([f_esm_base, f_esm_mut - f_esm_base])
                    
                    # 3. Biophysical & Structural Branch
                    f_protscale_base = np.array([self.protscale_df.get(aa, pd.Series(np.zeros(len(self.protscale_df)))).values for aa in seq_base], dtype=np.float32)
                    
                    try:
                        grantham_dist = self.grantham_df.loc[d['wt_aa'], d['mu_aa']]
                    except KeyError:
                        grantham_dist = 0.0
                        
                    protscale_wt_val = self.protscale_df.get(d['wt_aa'], pd.Series(np.zeros(len(self.protscale_df)))).values
                    protscale_mu_val = self.protscale_df.get(d['mu_aa'], pd.Series(np.zeros(len(self.protscale_df)))).values
                    protscale_sub = protscale_mu_val - protscale_wt_val
                    
                    f_mutation_profile = np.concatenate([[grantham_dist], protscale_wt_val, protscale_mu_val, protscale_sub])
                    f_mut_matrix = np.tile(f_mutation_profile, (seq_len, 1)).astype(np.float32)
                    
                    feat_physbio = np.hstack([f_protscale_base, f_geom, f_mut_matrix])

                    # ปรับความยาวเป็น 500 อะมิโน สำหรับแต่ละ Branch
                    tensor_prost = torch.tensor(self.get_fixed_length_500(feat_prost, mut_idx, target_len=500))
                    tensor_esm = torch.tensor(self.get_fixed_length_500(feat_esm, mut_idx, target_len=500))
                    tensor_physbio = torch.tensor(self.get_fixed_length_500(feat_physbio, mut_idx, target_len=500))

                    if not self.reported_shapes:
                        self.print_feature_report(tensor_prost, tensor_esm, tensor_physbio)
                        self.reported_shapes = True
                    
                    path = os.path.join(self.output_dir, f"{group}_{d['dir_name']}")
                    os.makedirs(path, exist_ok=True)
                    
                    # บันทึกแยกเป็น Dictionary เพื่อให้โหลดเข้าสถาปัตยกรรม Late Fusion ได้ทันที
                    file_name = f"{pdb_file.replace('.pdb','')}_{chain}_{wt}{pos}{mu}_{d['dir_name']}.pt"
                    file_path = os.path.join(path, file_name)
                    
                    feature_dict = {
                        'prost': tensor_prost,
                        'esm': tensor_esm,
                        'physbio': tensor_physbio
                    }
                    torch.save(feature_dict, file_path)

                    metadata_list.append({
                        'Tensor_File': file_name,
                        'Group': group,
                        'Direction': d['dir_name'],
                        'ddG': d['ddg']
                    })

                print(f"Success [{idx+1}/{len(df)}]: {wt}{pos}{mu} ({group})")

            except Exception as e:
                self.log_error(f"Skipping [{idx+1}]: error occured {e}")

        df_log = pd.DataFrame(metadata_list)
        df_log.to_csv('dataset_metadata_log.csv', index=False)

if __name__ == "__main__":
    pipeline = GranthamProtscaleLateFusionPipeline(device="cuda" if torch.cuda.is_available() else "cpu")
    pipeline.process_and_save_all("Data_for_train_model.csv")