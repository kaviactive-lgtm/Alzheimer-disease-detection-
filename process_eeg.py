import pandas as pd
import os

print("Starting EEG data processing...")

# --- Configuration ---
INPUT_EEG_FILE = os.path.join('data', 'eeg', 'AD_all_patients.csv')
OUTPUT_FEATURES_FILE = os.path.join('data', 'eeg_features.csv')

# These are the columns containing EEG sensor data
EEG_SENSOR_COLUMNS = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4']

try:
    # --- Load the data ---
    print(f"Loading raw EEG data from {INPUT_EEG_FILE}...")
    df = pd.read_csv(INPUT_EEG_FILE)
    print("Data loaded successfully.")

    # --- Group by Patient and calculate features ---
    # We group by the NEW 'PatientID' column and the 'status' column.
    # .agg() allows us to compute multiple statistics at once for each sensor.
    print("Grouping by PatientID and calculating features (mean, std)...")
    features_df = df.groupby(['PatientID', 'status'])[EEG_SENSOR_COLUMNS].agg(['mean', 'std']).reset_index()
    print("Feature calculation complete.")

    # --- Flatten the multi-level column headers ---
    # The headers will look like ('Fp1', 'mean'), ('Fp1', 'std'). We want 'Fp1_mean', 'Fp1_std'.
    features_df.columns = ['_'.join(col).strip() for col in features_df.columns.values]
    # The first two columns will be named 'PatientID_' and 'status_'. Let's fix them.
    features_df.rename(columns={'PatientID_': 'PatientID', 'status_': 'status'}, inplace=True)

    # --- Save the processed features ---
    features_df.to_csv(OUTPUT_FEATURES_FILE, index=False)

    print("\n" + "="*50)
    print("✅ EEG Processing Complete!")
    print(f"Saved processed features for {len(features_df)} patients to:")
    print(f"   {OUTPUT_FEATURES_FILE}")
    print("="*50)
    print("\nFirst 5 rows of the new features file:")
    print(features_df.head())


except FileNotFoundError:
    print(f"\n[ERROR] Input file not found: {INPUT_EEG_FILE}")
    print("Please make sure the file exists.")
except KeyError as e:
    print(f"\n[ERROR] A required column was not found: {e}")
    print("Please ensure you have added the 'PatientID' column and that the sensor names are correct.")