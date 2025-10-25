import pandas as pd
import os
import sys

# --- CONFIGURATION ---
# We now use the PROCESSED EEG features file.
CONFIG = {
    'eeg_features_file': os.path.join('data', 'eeg_features.csv'),
    'gene_file_path': os.path.join('data', 'gene_expression', 'GSE28379 - GSE28379_series_matrix.csv'),
    'output_metadata_path': os.path.join('data', 'metadata.csv'),
    'eeg_label_col': 'status', # The label column from eeg_features.csv
    'label_mapping': {'Control': 0, 'MCI': 1, 'AD': 2},
    'test_split_ratio': 0.2
}

def create_final_metadata():
    print("Creating final metadata from processed EEG features and gene data.")

    # --- Step 1: Read the processed EEG features ---
    try:
        eeg_df = pd.read_csv(CONFIG['eeg_features_file'])
        labels = eeg_df[CONFIG['eeg_label_col']].tolist()
        print(f"Found {len(eeg_df)} patients in the processed EEG file.")
    except FileNotFoundError:
        print(f"[ERROR] Processed EEG file not found: {CONFIG['eeg_features_file']}")
        print("Please run `process_eeg.py` first.")
        sys.exit()

    # --- Step 2: Extract GSM IDs from the Gene File Header ---
    try:
        with open(CONFIG['gene_file_path'], 'r') as f:
            header = f.readline().strip()
        gsm_ids = [col for col in header.split(',') if col.startswith('GSM')]
        print(f"Found {len(gsm_ids)} GSM samples in the gene file header.")
    except FileNotFoundError:
        print(f"[ERROR] Gene expression file not found: {CONFIG['gene_file_path']}")
        sys.exit()

    # --- Step 3: Check for Mismatch and Create Mapping ---
    if len(eeg_df) != len(gsm_ids):
        print("\n[CRITICAL WARNING] Number of patients in processed EEG file does not match gene samples!")
        sys.exit()

    metadata_records = []
    for gsm_id, text_label in zip(gsm_ids, labels):
        if text_label in CONFIG['label_mapping']:
            numeric_label = CONFIG['label_mapping'][text_label]
            metadata_records.append({'id': gsm_id, 'label': numeric_label})

    # --- Step 4: Create final DataFrame and add train/test split ---
    final_df = pd.DataFrame(metadata_records)
    # With only 4 samples, splitting doesn't make much sense, but we do it for code consistency.
    # A real project would need more data.
    final_df['split'] = ['train', 'train', 'test', 'test'] if len(final_df) == 4 else 'train'
    
    final_df.to_csv(CONFIG['output_metadata_path'], index=False)
    print(f"\n✅ Success! Final metadata file created at: {CONFIG['output_metadata_path']}")
    print(final_df)

if __name__ == '__main__':
    create_final_metadata()