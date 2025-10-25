import os

# The path to your image data
root_dir = os.path.join('data', 'smri', 'Combined Dataset')
output_file = 'all_smri_ids.txt'

all_ids = []

print(f"Scanning directory: {root_dir}")

# Walk through all subdirectories (train/Mild, train/Moderate, test/Mild, etc.)
for subdir, _, files in os.walk(root_dir):
    for filename in files:
        if filename.lower().endswith('.jpg'):
            # Get the filename without the .jpg extension
            base_name = os.path.splitext(filename)[0]
            all_ids.append(base_name)

# Save the list to a text file
with open(output_file, 'w') as f:
    for smri_id in sorted(all_ids):
        f.write(f"{smri_id}\n")

print(f"Found {len(all_ids)} image files.")
print(f"A complete list has been saved to: {output_file}")