import torch
import torch.nn as nn
import yaml
import logging
import argparse
import os
from torch.optim import AdamW
from torch.cuda import amp
import numpy as np

# --- Import all necessary modules ---
from src.data_loader import create_dataloaders
from src.components.hme_module import HMEModule
from src.components.caf_module import CAFModule
from src.components.dpgl_module import DPGLModule
from src import utils
from src import plotting

# =========================================================================
#  MODEL WRAPPERS FOR EACH TRAINING PHASE
# =========================================================================
class HME_Trainer(nn.Module):
    """Model specifically to train the HME component's encoders."""
    def __init__(self, config):
        super().__init__()
        self.hme = HMEModule(
            smri_embed_dim=config['MODEL']['HME']['SMRI_EMBED_DIM'],
            eeg_embed_dim=config['MODEL']['HME']['EEG_EMBED_DIM'],
            gene_embed_dim=config['MODEL']['HME']['GENE_EMBED_DIM']
        )
        # Classify by simple concatenation
        combined_dim = config['MODEL']['HME']['SMRI_EMBED_DIM'] + config['MODEL']['HME']['GENE_EMBED_DIM']
        self.classifier = nn.Linear(combined_dim, config['TRAINING']['NUM_CLASSES'])
    def forward(self, smri, gene):
        smri_feat = self.hme.smri_encoder(smri)
        gene_feat = self.hme.gene_encoder(gene)
        combined = torch.cat([smri_feat, gene_feat], dim=1)
        return self.classifier(combined)

class CAF_Trainer(nn.Module):
    """Model to train the CAF component, using a frozen, pre-trained HME."""
    def __init__(self, config, hme_checkpoint_path):
        super().__init__()
        # Load and freeze HME
        self.hme = HMEModule(
            smri_embed_dim=config['MODEL']['HME']['SMRI_EMBED_DIM'],
            eeg_embed_dim=config['MODEL']['HME']['EEG_EMBED_DIM'],
            gene_embed_dim=config['MODEL']['HME']['GENE_EMBED_DIM']
        )
        self.hme = utils.load_component_checkpoint(self.hme, hme_checkpoint_path)
        for param in self.hme.parameters(): param.requires_grad = False
        
        # Initialize trainable CAF
        self.caf = CAFModule(
            input_dim1=config['MODEL']['HME']['SMRI_EMBED_DIM'],
            input_dim2=config['MODEL']['HME']['GENE_EMBED_DIM'],
            attention_dim=config['MODEL']['CAF']['ATTENTION_DIM'],
            dropout_rate=config['MODEL']['DROPOUT_RATE']
        )
        fused_dim = config['MODEL']['HME']['SMRI_EMBED_DIM'] + config['MODEL']['HME']['GENE_EMBED_DIM']
        self.classifier = nn.Linear(fused_dim, config['TRAINING']['NUM_CLASSES'])
    def forward(self, smri, gene):
        with torch.no_grad():
            smri_feat = self.hme.smri_encoder(smri)
            gene_feat = self.hme.gene_encoder(gene)
        fused = self.caf(smri_feat, gene_feat)
        return self.classifier(fused)

class DPGL_Trainer(nn.Module):
    """Model to train the DPGL component, using a frozen, pre-trained HME."""
    def __init__(self, config, hme_checkpoint_path):
        super().__init__()
        # Load and freeze HME
        self.hme = HMEModule(
            smri_embed_dim=config['MODEL']['HME']['SMRI_EMBED_DIM'],
            eeg_embed_dim=config['MODEL']['HME']['EEG_EMBED_DIM'],
            gene_embed_dim=config['MODEL']['HME']['GENE_EMBED_DIM']
        )
        self.hme = utils.load_component_checkpoint(self.hme, hme_checkpoint_path)
        for param in self.hme.parameters(): param.requires_grad = False
        
        # Initialize trainable DPGL
        self.dpgl = DPGLModule(
            eeg_input_dim=config['MODEL']['HME']['SMRI_EMBED_DIM'], # Adapted to use sMRI dim
            gene_input_dim=config['MODEL']['HME']['GENE_EMBED_DIM'],
            graph_hidden_dim=config['MODEL']['DPGL']['GRAPH_HIDDEN_DIM'],
            output_dim=config['MODEL']['DPGL']['GRAPH_HIDDEN_DIM'],
            dropout_rate=config['MODEL']['DROPOUT_RATE']
        )
        self.classifier = nn.Linear(config['MODEL']['DPGL']['GRAPH_HIDDEN_DIM'], config['TRAINING']['NUM_CLASSES'])

    def forward(self, smri, gene):
        with torch.no_grad():
            smri_feat = self.hme.smri_encoder(smri)
            gene_feat = self.hme.gene_encoder(gene)
        refined = self.dpgl(smri_feat, gene_feat)
        return self.classifier(refined)

# =========================================================================
#  MAIN DRIVER FUNCTION
# =========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Modular training script for NeuroOmics-Net components.")
    parser.add_argument('component', type=str, choices=['HME', 'CAF', 'DPGL'], help="The component to train.")
    args = parser.parse_args()
    
    with open('config.yaml', 'r') as f: config = yaml.safe_load(f)
    
    # --- Setup component-specific paths from config ---
    component_name = args.component.lower()
    log_path = os.path.join(config['OUTPUT']['LOGS_DIR'], f'{component_name}_training.log')
    plot_dir = os.path.join(config['OUTPUT']['PLOTS_DIR'], component_name)
    trained_models_dir = config['OUTPUT']['TRAINED_MODELS_DIR']
    
    # *** MODIFIED: Define final save path as requested ***
    final_model_save_path = os.path.join(trained_models_dir, f'{component_name}_model.pth')
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(trained_models_dir, exist_ok=True)

    utils.setup_logging(log_path)
    utils.set_seed(config['TRAINING']['SET_SEED'])
    
    logging.info(f"--- Starting Training for: {args.component} Component ---")
    
    train_loader, val_loader = create_dataloaders(config)
    
    # *** MODIFIED: HME checkpoint path is now based on the new naming convention ***
    hme_checkpoint = os.path.join(trained_models_dir, 'hme_model.pth')

    if args.component == 'HME':
        model = HME_Trainer(config)
    elif args.component == 'CAF':
        if not os.path.exists(hme_checkpoint):
             raise FileNotFoundError(f"HME model not found at '{hme_checkpoint}'. Please train the HME component first by running 'python train_component.py HME'")
        model = CAF_Trainer(config, hme_checkpoint_path=hme_checkpoint)
    elif args.component == 'DPGL':
        if not os.path.exists(hme_checkpoint):
             raise FileNotFoundError(f"HME model not found at '{hme_checkpoint}'. Please train the HME component first by running 'python train_component.py HME'")
        model = DPGL_Trainer(config, hme_checkpoint_path=hme_checkpoint)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f"Model for {args.component} created and moved to {device}.")
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['TRAINING']['LEARNING_RATE'])
    loss_fn = nn.CrossEntropyLoss()
    scaler = amp.GradScaler(enabled=(device.type == 'cuda'))
    
    best_val_auc = 0.0
    best_model_state = None
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
    final_val_preds = None
    final_val_labels = None

    for epoch in range(config['TRAINING']['NUM_EPOCHS']):
        model.train()
        total_train_loss = 0
        for smri, gene, labels in train_loader:
            smri, gene, labels = smri.to(device), gene.to(device), labels.to(device)
            with torch.amp.autocast(device_type=device.type):
                outputs = model(smri, gene)
                loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()
        history['train_loss'].append(total_train_loss / len(train_loader))

        model.eval()
        total_val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for smri, gene, labels in val_loader:
                smri, gene, labels = smri.to(device), gene.to(device), labels.to(device)
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(smri, gene)
                    loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()
                all_preds.append(torch.softmax(outputs, dim=1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        history['val_loss'].append(total_val_loss / len(val_loader))
        
        y_true_epoch = np.concatenate(all_labels)
        y_pred_proba_epoch = np.concatenate(all_preds)
        val_metrics = utils.calculate_metrics(y_true_epoch, y_pred_proba_epoch, config['TRAINING']['NUM_CLASSES'])
        
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])
        
        logging.info(f"Epoch {epoch+1}/{config['TRAINING']['NUM_EPOCHS']} | Val Acc: {val_metrics['accuracy']:.2f}% | Val AUC: {val_metrics['auc']:.4f}")
        
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model_state = model.state_dict()
            final_val_preds = y_pred_proba_epoch
            final_val_labels = y_true_epoch
            logging.info(f"New best model found with AUC: {best_val_auc:.4f}")

    if best_model_state:
        component_to_save = None
        if args.component == 'HME':
            component_state_dict = {k.replace('hme.', ''): v for k, v in best_model_state.items() if k.startswith('hme.')}
            component_to_save = HMEModule(
                smri_embed_dim=config['MODEL']['HME']['SMRI_EMBED_DIM'],
                eeg_embed_dim=config['MODEL']['HME']['EEG_EMBED_DIM'],
                gene_embed_dim=config['MODEL']['HME']['GENE_EMBED_DIM']
            )
        elif args.component == 'CAF':
            component_state_dict = {k.replace('caf.', ''): v for k, v in best_model_state.items() if k.startswith('caf.')}
            component_to_save = CAFModule(
                input_dim1=config['MODEL']['HME']['SMRI_EMBED_DIM'],
                input_dim2=config['MODEL']['HME']['GENE_EMBED_DIM'],
                attention_dim=config['MODEL']['CAF']['ATTENTION_DIM'],
                dropout_rate=config['MODEL']['DROPOUT_RATE']
            )
        elif args.component == 'DPGL':
             component_state_dict = {k.replace('dpgl.', ''): v for k, v in best_model_state.items() if k.startswith('dpgl.')}
             component_to_save = DPGLModule(
                eeg_input_dim=config['MODEL']['HME']['SMRI_EMBED_DIM'],
                gene_input_dim=config['MODEL']['HME']['GENE_EMBED_DIM'],
                graph_hidden_dim=config['MODEL']['DPGL']['GRAPH_HIDDEN_DIM'],
                output_dim=config['MODEL']['DPGL']['GRAPH_HIDDEN_DIM'],
                dropout_rate=config['MODEL']['DROPOUT_RATE']
            )

        if component_to_save:
            component_to_save.load_state_dict(component_state_dict)
            # *** MODIFIED: Use the new save path ***
            utils.save_component_checkpoint(component_to_save, final_model_save_path)
            logging.info(f"--- Training for {args.component} complete. Best model saved to {final_model_save_path} ---")

    if final_val_labels is not None and final_val_preds is not None:
        logging.info(f"Generating plots for best {args.component} model...")
        plotting.plot_training_validation_curves(history, plot_dir, args.component)
        plotting.plot_confusion_matrix(final_val_labels, final_val_preds, config['DATA']['CLASS_NAMES'], plot_dir, args.component)
        plotting.plot_roc_curves(final_val_labels, final_val_preds, config['TRAINING']['NUM_CLASSES'], config['DATA']['CLASS_NAMES'], plot_dir, args.component)
        plotting.plot_precision_recall_curves(final_val_labels, final_val_preds, config['TRAINING']['NUM_CLASSES'], config['DATA']['CLASS_NAMES'], plot_dir, args.component)