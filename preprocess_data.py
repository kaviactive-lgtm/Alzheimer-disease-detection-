# /NEUROOMICS-NET/preprocess_data.py

import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# --- (1) CONFIGURATION ---
SMRI_ROOT_DIR = "data/smri/Combined Dataset"
GENE_CSV_PATH = "data/gene_expression/GSE28379 - GSE28379_series_matrix.csv"
ID_MAPPING_PATH = "id_mapping.csv"

OUTPUT_ROOT_DIR = "data_processed"
OUT_SMRI_DIR = os.path.join(OUTPUT_ROOT_DIR, "smri")
OUT_GENE_DIR = os.path.join(OUTPUT_ROOT_DIR, "gene_expression")
NEW_METADATA_PATH = os.path.join(OUTPUT_ROOT_DIR, "metadata.csv")

CLASS_MAPPING = {"No Impairment": 0, "Very Mild Impairment": 1, "Mild Impairment": 2, "Moderate Impairment": 3}
IMAGE_SIZE = (96, 96)
NUM_GENES_TO_SELECT = 1000 

def preprocess_with_mapping_debug():
    print(f"Creating/updating directory structure at '{OUTPUT_ROOT_DIR}'...")
    os.makedirs(OUT_SMRI_DIR, exist_ok=True); os.makedirs(OUT_GENE_DIR, exist_ok=True)

    # --- (2) LOAD MAPPING AND GENE DATA ---
    print("Loading ID mapping and Gene Expression CSVs...")
    try:
        id_map_df = pd.read_csv(ID_MAPPING_PATH)
        id_lookup = dict(zip(id_map_df.sMRI_ID, id_map_df.Gene_ID))
        
        gene_df = pd.read_csv(GENE_CSV_PATH, comment='!').set_index('ID_REF').T
        gene_df.index.name = 'Gene_ID'
    except Exception as e:
        print(f"[FATAL ERROR] Could not process a file: {e}"); return

    # --- (3) DEBUGGING: PRINT IDs ---
    print("\n--- DEBUGGING ID MISMATCH ---")
    print(f"First 5 sMRI_IDs from '{ID_MAPPING_PATH}': {list(id_lookup.keys())[:5]}")
    print(f"First 5 Gene_IDs from '{ID_MAPPING_PATH}': {list(id_lookup.values())[:5]}")
    print(f"First 5 REAL Gene IDs from the gene file index: {list(gene_df.index)[:5]}")
    print("---------------------------------\n")
    print("Carefully compare the 'Gene_IDs from id_mapping.csv' with the 'REAL Gene IDs'. They must match EXACTLY.")
    
    new_metadata_records = []
    
    # --- (4) ITERATE, MATCH, AND PROCESS ---
    for split_type in ["train", "test"]:
        print(f"\nProcessing '{split_type}' set...")
        split_dir = os.path.join(SMRI_ROOT_DIR, split_type)
        
        for class_name, label in CLASS_MAPPING.items():
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir): continue

            for image_filename in tqdm(os.listdir(class_dir), desc=f"  - Class: {class_name}"):
                smri_id = os.path.splitext(image_filename)[0]
                gene_id = id_lookup.get(smri_id)
                
                if not gene_id:
                    continue

                if gene_id not in gene_df.index:
                    print(f"\n[Debug Warning] Match failed for sMRI ID '{smri_id}'.")
                    print(f"  - Your mapping file provided Gene ID: '{gene_id}'")
                    print(f"  - This ID was NOT FOUND in the gene data file.")
                    continue

                try:
                    # --- If we get here, a match was successful! ---
                    gene_features = gene_df.loc[gene_id].values.astype(np.float32)[:NUM_GENES_TO_SELECT]
                    image_path = os.path.join(class_dir, image_filename)
                    img = Image.open(image_path).convert('L'); img_resized = img.resize(IMAGE_SIZE)
                    smri_array = np.array(img_resized, dtype=np.float32) / 255.0

                    final_patient_id = str(gene_id)
                    np.save(os.path.join(OUT_SMRI_DIR, f"{final_patient_id}_smri.npy"), smri_array)
                    np.save(os.path.join(OUT_GENE_DIR, f"{final_patient_id}_gene.npy"), gene_features)
                    
                    new_metadata_records.append({"id": final_patient_id, "label": label, "split": split_type})
                except Exception as e:
                    print(f"  [Error] for sMRI ID '{smri_id}': {e}")
    
    print("\nSaving new master metadata file...")
    final_metadata_df = pd.DataFrame(new_metadata_records)
    final_metadata_df.to_csv(NEW_METADATA_PATH, index=False)
    
    print("="*50)
    print(" Bimodal (sMRI + Gene) PREPROCESSING COMPLETE!")
    print(f" {len(final_metadata_df)} samples with matching IDs were processed.")
    print("="*50)

if __name__ == '__main__':
    preprocess_with_mapping_debug()