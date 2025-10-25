import pandas as pd
import os

# This script is designed to safely inspect your EEG data file
# to find out why the processing is failing.

print("--- EEG File Diagnostic Script ---")

# --- Configuration ---
EEG_FILE_PATH = os.path.join('data', 'eeg', 'AD_all_patients.csv')
EXPECTED_ID_COLUMN = 'PatientID'

try:
    # --- 1. Load the data ---
    print(f"\n[INFO] Attempting to load the file: {EEG_FILE_PATH}")
    df = pd.read_csv(EEG_FILE_PATH)
    print("[SUCCESS] File loaded successfully into pandas.")

    # --- 2. Check the column headers ---
    print("\n[INFO] Checking column headers...")
    actual_columns = df.columns.tolist()
    print(f"   > Columns found: {actual_columns}")

    if EXPECTED_ID_COLUMN in actual_columns:
        print(f"   > [SUCCESS] The '{EXPECTED_ID_COLUMN}' column exists.")
    else:
        print(f"   > [ERROR] The '{EXPECTED_ID_COLUMN}' column was NOT FOUND!")
        print("     Please check for typos, capitalization, or extra spaces in your CSV header.")
        exit() # Stop the script if the column doesn't exist

    # --- 3. Inspect the first 5 rows ---
    print(f"\n[INFO] Displaying the first 5 rows of the file to check data:")
    print(df.head())

    # --- 4. Analyze the PatientID column specifically ---
    print(f"\n[INFO] Analyzing the content of the '{EXPECTED_ID_COLUMN}' column...")
    
    # Count how many rows actually have a value in the PatientID column
    non_empty_count = df[EXPECTED_ID_COLUMN].notna().sum()
    
    print(f"   > Total rows in file: {len(df)}")
    print(f"   > Number of rows with a non-empty PatientID: {non_empty_count}")

    # Show the unique values found in the column
    unique_values = df[EXPECTED_ID_COLUMN].unique()
    print(f"   > Unique values found in PatientID column: {unique_values}")
    
    # --- 5. Final Conclusion ---
    print("\n--- CONCLUSION ---")
    if non_empty_count > 0:
        print("✅ [GOOD NEWS] The script found data in your 'PatientID' column!")
        print("   This means the file was likely saved correctly. You should now be able to run 'process_eeg.py' successfully.")
    else:
        print("❌ [PROBLEM FOUND] The 'PatientID' column exists, but it appears to be completely empty.")
        print("   This is why you are getting 0 patients.")
        print("\n   ACTION: Please re-open 'AD_all_patients.csv' and manually fill the 'PatientID' column with values (e.g., 'P1', 'P2', 'P3', 'P4') and save the file again.")


except FileNotFoundError:
    print(f"\n[ERROR] The file was not found at the expected location: {EEG_FILE_PATH}")
except Exception as e:
    print(f"\n[UNEXPECTED ERROR] An error occurred: {e}")