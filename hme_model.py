print("--- Part 1: Initializing Environment and Importing Libraries ---")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchinfo import summary
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize, StandardScaler
from itertools import cycle
from tqdm import tqdm
import os
import wfdb # Library for reading PhysioNet data
from imblearn.over_sampling import SMOTE
from scipy.ndimage import gaussian_filter1d

print("Libraries imported successfully.")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Execution device configured to: {DEVICE}")

# --- Configuration ---
n_classes = 5
EPOCHS = 15 # Kept the same as simulation for consistency
BATCH_SIZE = 128
LEARNING_RATE = 0.001
DB_NAME = 'mitdb' # MIT-BIH Arrhythmia Database
RESULTS_DIR = 'HME_Model_results' # ADDED: Directory to save all outputs

# ADDED: Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Results will be saved in '{RESULTS_DIR}/' directory.")


# ===================================================================
# Part 2: Loading and Pre-processing Datasets for HME_Model
# ===================================================================
print("\n--- Part 2: Loading and Pre-processing Datasets for HME_Model ---")

# Define AAMI classes mapping
aami_classes = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal (N)
    'A': 1, 'a': 1, 'J': 1, 'S': 1,          # Supraventricular Ectopic (S)
    'V': 2, 'E': 2,                          # Ventricular Ectopic (V)
    'F': 3,                                  # Fusion (F)
    'P': 4, 'f': 4, 'U': 4, '/': 4, 'Q': 4,  # Unknown (Q)
}
mit_classes = ['N', 'S', 'V', 'F', 'Q'] # Re-ordered to match mapping

def load_mit_bih_data():
    """Downloads, parses, and preprocesses the MIT-BIH dataset."""
    if not os.path.exists(DB_NAME):
        print(f"Downloading MIT-BIH dataset ('{DB_NAME}')...")
        wfdb.dl_database(DB_NAME, os.getcwd())
        print("Download complete.")

    records = wfdb.get_record_list(DB_NAME)
    excluded_records = ['102', '104', '107', '217']
    records = [r for r in records if r not in excluded_records]

    all_signals = []
    all_labels = []

    print("Processing records to extract heartbeats...")
    for rec_name in tqdm(records):
        record = wfdb.rdrecord(os.path.join(DB_NAME, rec_name))
        annotation = wfdb.rdann(os.path.join(DB_NAME, rec_name), 'atr')
        signal = record.p_signal[:, 0]
        symbols = annotation.symbol
        beat_locations = annotation.sample
        for i, location in enumerate(beat_locations):
            symbol = symbols[i]
            if symbol in aami_classes:
                start, end = location - 100, location + 180
                if start >= 0 and end < len(signal):
                    beat_signal = signal[start:end]
                    all_signals.append(beat_signal)
                    all_labels.append(aami_classes[symbol])
    return np.array(all_signals), np.array(all_labels)

X, y = load_mit_bih_data()
print(f"Loaded {len(X)} total heartbeats from MIT-BIH.")

# --- Data Splitting and Scaling ---
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(f"Original training set distribution: {np.bincount(y_train)}")

# --- Apply SMOTE ---
print("Applying SMOTE to balance the training dataset...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"New resampled training set size: {len(X_train_resampled)} samples.")
print(f"Resampled training set distribution: {np.bincount(y_train_resampled)}")

# Reshape data for the model
X_train_resampled = X_train_resampled[:, :, np.newaxis]
X_test = X_test[:, :, np.newaxis]

# --- Create PyTorch Datasets and DataLoaders ---
class ECGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset_orig = ECGDataset(X_train[:, :, np.newaxis], y_train)
val_size = int(0.2 * len(train_dataset_orig))
train_size_final = len(train_dataset_orig) - val_size
_, val_dataset = random_split(train_dataset_orig, [train_size_final, val_size])

train_dataset_resampled = ECGDataset(X_train_resampled, y_train_resampled)
test_dataset = ECGDataset(X_test, y_test)
train_loader = DataLoader(train_dataset_resampled, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===================================================================
# Part 3: Defining the HME_Model Architecture
# ===================================================================
print("\n--- Part 3: Defining the HME_Model Architecture ---")

class HME_ModelClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(HME_ModelClassifier, self).__init__()
        self.recurrent_layer = nn.LSTM(input_size, hidden_size, num_layers,
                                       batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.recurrent_layer(x)
        h_n_forward = h_n[-2,:,:]
        h_n_backward = h_n[-1,:,:]
        h_n_cat = torch.cat((h_n_forward, h_n_backward), dim=1)
        out = self.dropout(h_n_cat)
        out = self.fc(out)
        return out

input_size = X_train_resampled.shape[2]
hidden_size = 128
num_layers = 2
model = HME_ModelClassifier(input_size, hidden_size, num_layers, n_classes).to(DEVICE)
print("\nHME_Model successfully initialized. Displaying summary:")
summary(model, input_size=(BATCH_SIZE, X_train_resampled.shape[1], input_size))

# ===================================================================
# Part 4: HME_Model Training Phase
# ===================================================================
print("\n--- Part 4: HME_Model Training Phase ---")
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct_preds, total_preds = 0.0, 0, 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()
    train_loss = running_loss / total_preds
    train_acc = correct_preds / total_preds
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)

    model.eval()
    running_loss, correct_preds, total_preds = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()
    val_loss = running_loss / total_preds
    val_acc = correct_preds / total_preds
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    scheduler.step()
    print(f"Epoch {epoch+1}/{EPOCHS} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

print("\n--- Training Complete ---")

# ===================================================================
# Part 5: Evaluation and Visualizations
# ===================================================================
print("\n--- Part 5: Evaluation and Visualizations ---")

# --- Accuracy and Loss Curves ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(history['train_acc'], label='Train Accuracy', color='blue', marker='o')
ax1.plot(history['val_acc'], label='Validation Accuracy', color='orange', marker='o')
ax1.set_title('HME_Model Accuracy'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
ax1.legend(); ax1.grid(True)
ax2.plot(history['train_loss'], label='Train Loss', color='blue', marker='o')
ax2.plot(history['val_loss'], label='Validation Loss', color='orange', marker='o')
ax2.set_title('HME_Model Loss'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
ax2.legend(); ax2.grid(True)
plt.suptitle("Model Training History")
plt.savefig(os.path.join(RESULTS_DIR, "hme_model_training_history.png")) # ADDED
plt.show()

# --- Final Evaluation on Test Set ---
print("\n--- Evaluating on the held-out Test Set ---")
model.eval()
y_true, y_pred, y_prob = [], [], []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        probabilities = nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())
        y_prob.extend(probabilities.cpu().numpy())
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

# --- MODIFIED: Capture report as a string and save to file ---
report_str = classification_report(y_true, y_pred, target_names=mit_classes, zero_division=0)
print("\nClassification Report:")
print(report_str)

# Save the report to a text file
with open(os.path.join(RESULTS_DIR, "hme_model_classification_report.txt"), 'w') as f:
    f.write("Classification Report\n\n")
    f.write(report_str)
print(f"Classification report saved to {os.path.join(RESULTS_DIR, 'hme_model_classification_report.txt')}")


# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=mit_classes, yticklabels=mit_classes)
plt.title('HME_Model Confusion Matrix'); plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.savefig(os.path.join(RESULTS_DIR, "hme_model_confusion_matrix.png")) # ADDED
plt.show()

# --- ROC Curve ---
y_true_bin = label_binarize(y_true, classes=range(n_classes))
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    if np.sum(y_true_bin[:, i]) > 0:
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
colors = cycle(['purple', 'magenta', 'dodgerblue', 'teal', 'saddlebrown'])
for i, color in zip(range(n_classes), colors):
    if i in roc_auc:
        plt.plot(fpr[i], tpr[i], color=color, lw=2.5,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(mit_classes[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('HME_Model Multi-class ROC Curve'); plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "hme_model_roc_curve.png")) # ADDED
plt.show()

# --- Additional Visualization 1: Learning Rate ---
lr_schedule = [LEARNING_RATE * (0.5 ** (i // 5)) for i in range(EPOCHS)]
plt.figure(figsize=(8, 4))
plt.plot(lr_schedule, marker='o', color='green')
plt.title('Learning Rate per Epoch'); plt.xlabel('Epoch'); plt.ylabel('Learning Rate')
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "learning_rate_schedule.png")) # ADDED
plt.show()

# --- Additional Visualization 2: Accuracy vs Loss Correlation ---
plt.figure(figsize=(8, 6))
plt.scatter(history['train_loss'], history['train_acc'], label='Train', c='blue', alpha=0.7)
plt.scatter(history['val_loss'], history['val_acc'], label='Validation', c='orange', alpha=0.7)
plt.title('Accuracy vs. Loss')
plt.xlabel('Loss'); plt.ylabel('Accuracy')
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_vs_loss_correlation.png")) # ADDED
plt.show()

# --- Additional Visualization 3: Smoothed Accuracy & Loss (Trendline) ---
train_acc_smooth = gaussian_filter1d(history['train_acc'], sigma=1)
val_acc_smooth = gaussian_filter1d(history['val_acc'], sigma=1)
train_loss_smooth = gaussian_filter1d(history['train_loss'], sigma=1)
val_loss_smooth = gaussian_filter1d(history['val_loss'], sigma=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(train_acc_smooth, label='Train Accuracy (Smoothed)', color='blue')
ax1.plot(val_acc_smooth, label='Validation Accuracy (Smoothed)', color='orange')
ax1.set_title('Smoothed Accuracy'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
ax1.legend(); ax1.grid(True)
ax2.plot(train_loss_smooth, label='Train Loss (Smoothed)', color='blue')
ax2.plot(val_loss_smooth, label='Validation Loss (Smoothed)', color='orange')
ax2.set_title('Smoothed Loss'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
ax2.legend(); ax2.grid(True)
plt.suptitle("Smoothed Training Trends")
plt.savefig(os.path.join(RESULTS_DIR, "smoothed_training_trends.png")) # ADDED
plt.show()

# --- Additional Visualization 4: Epoch-wise Training Stability ---
diff_acc = np.array(history['train_acc']) - np.array(history['val_acc'])
plt.figure(figsize=(8, 5))
plt.plot(diff_acc, color='red', marker='o')
plt.axhline(y=0, color='gray', linestyle='--')
plt.title('Generalization Gap (Train Acc - Val Acc)')
plt.xlabel('Epoch'); plt.ylabel('Accuracy Gap')
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "generalization_gap.png")) # ADDED
plt.show()

# --- Additional Visualization 5: Bar Chart of Final Class-wise Metrics ---
# MODIFIED: Added labels parameter to handle missing classes if any
report = classification_report(y_true, y_pred, target_names=mit_classes, output_dict=True, labels=range(n_classes), zero_division=0)
df_metrics = pd.DataFrame(report).transpose().loc[mit_classes]

df_metrics[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(12, 7), colormap='viridis')
plt.title('Final Evaluation Metrics by Class on Test Set')
plt.ylabel('Score'); plt.ylim([0, 1.05])
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.savefig(os.path.join(RESULTS_DIR, "class_metrics_barchart.png")) # ADDED
plt.show()