"""
Graph Neural Network (GNN) Model for EGFR Bioactivity Prediction
===============================================================

This script defines and trains an advanced Graph Neural Network (GNN) model to predict
the bioactivity (specifically, the IC50 value) of chemical compounds against the
Epidermal Growth Factor Receptor (EGFR), a critical target in cancer research.

The script is a complete pipeline for GNN-based drug discovery, including:
1.  **Molecular Featurization**: It converts molecules from their SMILES string representation
    into detailed graph structures. Each atom becomes a node with rich chemical features,
    and each bond becomes an edge with its own set of features. This is handled by the
    `EnhancedMolecularProcessor` class.
2.  **GNN Model Architecture**: It defines a sophisticated GNN architecture, `AdvancedEGFRNet`,
    which uses Graph Attention (GAT) layers. GAT layers allow the model to learn the
    relative importance of different atoms and bonds within a molecule when making a prediction,
    which is a powerful way to capture complex chemical relationships.
3.  **Data Loading and Preprocessing**: The `LargeDatasetLoader` class handles the loading of
    the training data, converting molecules to graphs, and splitting the data into
    training, validation, and test sets. This ensures that the model is trained and
    evaluated robustly.
4.  **Model Training**: The `EnhancedTrainer` class manages the entire training process. It
    includes modern deep learning techniques such as:
    - AdamW optimizer and a learning rate scheduler for efficient training.
    - Gradient clipping to prevent unstable training.
    - Early stopping to prevent overfitting and save the best version of the model.
5.  **Evaluation and Visualization**: After training, the script evaluates the model's
    performance on the held-out test set, calculating key regression and classification
    metrics. It also generates a comprehensive set of plots to visualize the model's
    performance, such as training curves, scatter plots of predicted vs. actual values,
    and a confusion matrix.

This script is the core of the predictive modeling part of the project, creating the
intelligent engine that powers the subsequent drug discovery screening.
"""

# --- Core Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --- Scikit-learn for machine learning utilities ---
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import StandardScaler

# --- PyTorch and PyTorch Geometric for GNNs ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool, BatchNorm, LayerNorm

# --- RDKit for Cheminformatics ---
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    print("✅ RDKit imported successfully")
except ImportError:
    print("❌ RDKit not found. This script requires RDKit for molecule processing.")
    exit()

# Ignore minor warnings for a cleaner output
warnings.filterwarnings('ignore')

# =====================================================================================
#  1. Molecular Processing: Converting Molecules to Graphs
# =====================================================================================

class EnhancedMolecularProcessor:
    """
    A class responsible for converting a molecule's SMILES string into a detailed
    graph representation suitable for a GNN. This involves creating feature vectors
    for both atoms (nodes) and bonds (edges).
    """
    
    def __init__(self):
        """Initializes the processor, including pre-computing some chemical properties."""
        # A dictionary mapping Pauling electronegativity values to atomic numbers for common elements.
        self.electronegativity = {
            1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 15: 2.19, 16: 2.58, 
            17: 3.16, 35: 2.96, 53: 2.66, 5: 2.04, 14: 1.90, 33: 2.18, 34: 2.55
        }
        
    def atom_to_features(self, atom: Chem.Atom) -> np.ndarray:
        """
        Generates a comprehensive feature vector for a single atom.

        Args:
            atom (rdkit.Chem.Atom): The RDKit atom object.

        Returns:
            np.ndarray: A NumPy array representing the atom's features.
        """
        features = []
        
        # --- Atomic and Connectivity Features ---
        atomic_num = atom.GetAtomicNum()
        # One-hot encoding for the most common elements (Hydrogen to Calcium).
        features.extend([1 if atomic_num == x else 0 for x in range(1, 21)])
        # A normalized value for less common elements.
        features.append(atomic_num / 100.0)
        
        degree = atom.GetDegree() # Number of bonds connected to the atom.
        features.extend([1 if degree == x else 0 for x in range(7)]) # One-hot encoding for degree.
        features.append(degree / 6.0)  # Normalized degree.
        
        # --- Chemical Property Features ---
        features.append(atom.GetFormalCharge())
        features.append(atom.GetIsAromatic())
        features.append(atom.IsInRing())
        
        try:
            hyb = int(atom.GetHybridization())
            features.extend([1 if hyb == x else 0 for x in range(8)]) # One-hot encoding for hybridization type.
        except:
            features.extend([0] * 8)
        
        features.append(atom.GetTotalNumHs()) # Number of connected hydrogens.
        
        try:
            features.append(atom.GetValence(getExplicit=False)) # The atom's valence.
        except:
            features.append(0)
        
        features.append(self.electronegativity.get(atomic_num, 2.0)) # Electronegativity.
        
        # --- Ring Features ---
        features.append(atom.IsInRingSize(3))
        features.append(atom.IsInRingSize(4))
        features.append(atom.IsInRingSize(5))
        features.append(atom.IsInRingSize(6))
        features.append(atom.IsInRingSize(7))
        
        # --- Domain-specific Features ---
        features.append(1 if atomic_num in [6, 7, 8, 16] else 0)  # Is it a common biological atom?
        features.append(1 if atomic_num in [9, 17, 35, 53] else 0)  # Is it a halogen?
        
        return np.array(features, dtype=np.float32)
    
    def bond_to_features(self, bond: Chem.Bond) -> np.ndarray:
        """
        Generates a feature vector for a single bond.

        Args:
            bond (rdkit.Chem.Bond): The RDKit bond object.

        Returns:
            np.ndarray: A NumPy array representing the bond's features.
        """
        features = []
        
        bond_type = bond.GetBondTypeAsDouble()
        # One-hot encoding for bond type (single, aromatic, double, triple).
        features.extend([1 if bond_type == x else 0 for x in [1.0, 1.5, 2.0, 3.0]])
        
        features.append(bond.GetIsAromatic())
        features.append(bond.GetIsConjugated())
        features.append(bond.IsInRing())
        
        try:
            stereo = int(bond.GetStereo())
            features.extend([1 if stereo == x else 0 for x in range(6)]) # One-hot encoding for stereochemistry.
        except:
            features.extend([0] * 6)
        
        return np.array(features, dtype=np.float32)
    
    def smiles_to_graph(self, smiles: str) -> Data:
        """
        Converts a SMILES string into a PyTorch Geometric `Data` object (a graph).

        Args:
            smiles (str): The SMILES string representation of the molecule.

        Returns:
            torch_geometric.data.Data: A graph object for the GNN, or None if conversion fails.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return None
            
            # Explicitly add hydrogens to the molecule for a more complete graph representation.
            mol = Chem.AddHs(mol)
            
            # Create atom (node) features.
            atom_features = [self.atom_to_features(atom) for atom in mol.GetAtoms()]
            x = torch.tensor(atom_features, dtype=torch.float)
            
            # Create bond (edge) features and connectivity information.
            edge_indices, edge_features = [], []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bond_feat = self.bond_to_features(bond)
                
                # Edges in PyG are represented as a [2, num_edges] tensor.
                # We add the bond in both directions to represent an undirected graph.
                edge_indices.extend([[i, j], [j, i]])
                edge_features.extend([bond_feat, bond_feat])
            
            if edge_indices:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_features, dtype=torch.float)
            else:
                # Handle molecules with no bonds (e.g., single atoms).
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, len(edge_features[0]) if edge_features else 10), dtype=torch.float)
            
            # Return the final graph object.
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return None

# =====================================================================================
#  2. GNN Model Architecture
# =====================================================================================

class AdvancedEGFRNet(nn.Module):
    """
    An advanced Graph Neural Network for predicting EGFR bioactivity.
    This model uses Graph Attention (GAT) layers, which are particularly effective for
    learning from molecular graphs.
    """
    
    def __init__(self, num_node_features, num_edge_features, hidden_dim=256, num_layers=4, 
                 num_heads=8, dropout=0.1, use_batch_norm=True):
        super(AdvancedEGFRNet, self).__init__()
        
        # --- Input Layers ---
        # These layers project the initial node and edge features into a higher-dimensional space.
        self.node_embedding = nn.Linear(num_node_features, hidden_dim)
        self.edge_embedding = nn.Linear(num_edge_features, hidden_dim)
        
        # --- GAT Layers ---
        # The core of the model. We create a stack of GAT layers.
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            # A GAT layer that considers both node and edge features.
            conv = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                           dropout=dropout, edge_dim=hidden_dim, concat=True)
            self.gat_layers.append(conv)
            # Batch normalization helps stabilize training.
            self.batch_norms.append(BatchNorm(hidden_dim) if use_batch_norm else nn.Identity())
        
        # Dropout layer for regularization.
        self.dropout = nn.Dropout(dropout)
        
        # --- Output Layers ---
        # After processing the graph, we need to get a single vector representation for the whole molecule.
        # This is done by "pooling" the node features. We use three different types of pooling
        # and concatenate them for a richer representation.
        self.pool_dim = hidden_dim * 3
        
        # The final prediction head is a multi-layer perceptron (MLP) that maps the graph
        # representation to a single output value (the predicted log_activity).
        self.prediction_head = nn.Sequential(
            nn.Linear(self.pool_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 4, 1) # Final output layer.
        )
        
        # Initialize the model's weights with a good starting strategy.
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initializes the weights of linear layers for better training stability."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, data):
        """The forward pass of the model."""
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 1. Embed initial features.
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr) if edge_attr.size(0) > 0 else None
        
        # 2. Pass through GAT layers with residual connections.
        for gat, bn in zip(self.gat_layers, self.batch_norms):
            residual = x # Save the input for the residual connection.
            
            x = gat(x, edge_index, edge_attr)
            
            x = bn(x) if x.size(0) > 1 else x
            x = F.relu(x)
            x = self.dropout(x)
            
            # A residual connection helps with training deeper networks.
            if residual.shape == x.shape:
                x = x + residual
        
        # 3. Global Pooling.
        # This aggregates node features into a single graph-level feature vector.
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_add = global_add_pool(x, batch)
        
        x = torch.cat([x_mean, x_max, x_add], dim=1)
        
        # 4. Final Prediction.
        x = self.prediction_head(x)
        
        return x.squeeze()

# =====================================================================================
#  3. Data Loading and Splitting
# =====================================================================================

class LargeDatasetLoader:
    """Handles the loading, processing, and splitting of the dataset."""
    
    def __init__(self, csv_file, processor, test_size=0.2, val_size=0.1, random_state=42):
        self.csv_file = csv_file
        self.processor = processor
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
    
    def load_and_process_data(self):
        """Main method to load, process, and split the data."""
        
        print("📊 LOADING LARGE DATASET")
        print("=" * 30)
        df = pd.read_csv(self.csv_file)
        print(f"✅ Loaded {len(df)} compounds")
        
        # Convert all molecules in the DataFrame to graph objects.
        print("🧬 Converting molecules to graphs...")
        graphs, activities, names = self._molecules_to_graphs(df)
        
        # Create binary labels (Active/Inactive) for stratified splitting.
        binary_labels = torch.tensor([1 if act <= np.log10(1000) else 0 for act in activities], dtype=torch.float)
        
        # Split the data into training+validation and test sets.
        # Stratified splitting ensures that the proportion of active/inactive compounds is the same in both sets.
        train_val_graphs, test_graphs, train_val_activities, test_activities, \
        train_val_binary, test_binary, train_val_names, test_names = train_test_split(
            graphs, activities, binary_labels, names, 
            test_size=self.test_size, random_state=self.random_state, stratify=binary_labels
        )
        
        # Further split the training+validation set into separate training and validation sets.
        train_graphs, val_graphs, train_activities, val_activities, \
        train_binary, val_binary, train_names, val_names = train_test_split(
            train_val_graphs, train_val_activities, train_val_binary, train_val_names,
            test_size=self.val_size/(1-self.test_size), random_state=self.random_state, 
            stratify=train_val_binary
        )
        
        print(f"📊 Dataset split:")
        print(f"   Training: {len(train_graphs)} compounds")
        print(f"   Validation: {len(val_graphs)} compounds") 
        print(f"   Test: {len(test_graphs)} compounds")
        
        return (train_graphs, train_activities, train_binary, train_names,
                val_graphs, val_activities, val_binary, val_names,
                test_graphs, test_activities, test_binary, test_names)
    
    def _molecules_to_graphs(self, df):
        """Helper function to convert a DataFrame of molecules to a list of graph objects."""
        graphs, activities, names = [], [], []
        total, failed_count = len(df), 0
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"   Processing {idx+1}/{total} ({(idx+1)/total*100:.1f}%)")
            
            smiles = row['canonical_smiles']
            activity = row['log_activity']
            name = row.get('compound_name', f'Compound_{idx}')
            
            graph = self.processor.smiles_to_graph(smiles)
            
            if graph is not None:
                graphs.append(graph)
                activities.append(activity)
                names.append(str(name) if pd.notna(name) else f'Unknown_{idx}')
            else:
                failed_count += 1
        
        print(f"✅ Successfully processed {len(graphs)} molecules")
        print(f"⚠️ Failed to process {failed_count} molecules")
        
        return graphs, torch.tensor(activities, dtype=torch.float), names

# =====================================================================================
#  4. Model Training
# =====================================================================================

class EnhancedTrainer:
    """Manages the model training and validation loop."""
    
    def __init__(self, model, train_loader, val_loader, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # We use the AdamW optimizer, an improved version of the standard Adam optimizer.
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        # The learning rate scheduler reduces the learning rate automatically if the validation loss stops improving.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        # The loss function for this regression task is Mean Squared Error (MSE).
        self.criterion = nn.MSELoss()
        
        # Variables for tracking training progress and implementing early stopping.
        self.train_losses, self.val_losses = [], []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 20 # Stop training if validation loss doesn't improve for 20 epochs.
    
    def train_epoch(self):
        """Performs a single training epoch."""
        self.model.train() # Set the model to training mode.
        total_loss = 0
        
        for batch_data, batch_targets, _ in self.train_loader:
            self.optimizer.zero_grad()
            
            predictions = self.model(batch_data)
            loss = self.criterion(predictions, batch_targets)
            
            loss.backward()
            # Gradient clipping helps prevent exploding gradients, a common issue in training deep networks.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Performs a single validation epoch."""
        self.model.eval() # Set the model to evaluation mode.
        total_loss = 0
        
        with torch.no_grad(): # We don't need to calculate gradients during validation.
            for batch_data, batch_targets, _ in self.val_loader:
                predictions = self.model(batch_data)
                loss = self.criterion(predictions, batch_targets)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs=100):
        """The main training loop."""
        print(f"🚀 ENHANCED TRAINING STARTED")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            # --- Early Stopping Logic ---
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save the model only when it has improved on the validation set.
                torch.save(self.model.state_dict(), 'egfr_model.pth')
            else:
                self.patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            if self.patience_counter >= self.max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load the best performing model at the end of training.
        self.model.load_state_dict(torch.load('egfr_model.pth'))
        print("✅ Training completed!")
        
        return self.train_losses, self.val_losses

# =====================================================================================
#  5. Main Execution Block
# =====================================================================================

def main_large_dataset_training():
    """The main function to orchestrate the entire training pipeline."""
    
    print("🧠 ENHANCED EGFR DRUG DISCOVERY - LARGE SCALE TRAINING")
    print("=" * 55)
    
    # Check if the required dataset file exists.
    import glob
    dataset_files = glob.glob('egfr_dataset_*_compounds.csv')
    if not dataset_files:
        print("❌ No dataset found! Please run dataset_download.py first.")
        return
    
    # Use the largest dataset found.
    dataset_file = max(dataset_files, key=os.path.getsize)
    print(f"📊 Using dataset: {dataset_file}")
    
    # --- Data Loading ---
    processor = EnhancedMolecularProcessor()
    data_loader = LargeDatasetLoader(dataset_file, processor)
    (train_graphs, train_activities, _, _,
     val_graphs, val_activities, _, _,
     test_graphs, test_activities, _, test_names) = data_loader.load_and_process_data()
    
    # --- Dataset and DataLoader Creation ---
    from torch.utils.data import Dataset, DataLoader as TorchDataLoader
    
    class MolDataset(Dataset):
        def __init__(self, graphs, targets, names):
            self.graphs, self.targets, self.names = graphs, targets, names
        def __len__(self): return len(self.graphs)
        def __getitem__(self, idx): return self.graphs[idx], self.targets[idx], self.names[idx]
    
    train_dataset = MolDataset(train_graphs, train_activities, [""]*len(train_graphs))
    val_dataset = MolDataset(val_graphs, val_activities, [""]*len(val_graphs))
    test_dataset = MolDataset(test_graphs, test_activities, test_names)
    
    # --- Model Initialization ---
    sample_graph = train_graphs[0]
    num_node_features = sample_graph.x.shape[1]
    num_edge_features = sample_graph.edge_attr.shape[1]
    
    model = AdvancedEGFRNet(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=256, num_layers=4, num_heads=8, dropout=0.1
    )
    
    print(f"🧠 Model architecture initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")
    
    # Use a custom collate function required by PyTorch for batching graph data.
    def collate_fn(batch):
        graphs, targets, names = zip(*batch)
        from torch_geometric.data import Batch
        batched_graphs = Batch.from_data_list(graphs)
        return batched_graphs, torch.stack(list(targets)), names

    train_loader = TorchDataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = TorchDataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = TorchDataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # --- Training ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    trainer = EnhancedTrainer(model, train_loader, val_loader, device)
    train_losses, val_losses = trainer.train(num_epochs=100)
    
    # --- Final Evaluation ---
    print("\n📊 FINAL MODEL EVALUATION ON TEST SET")
    print("=" * 35)
    
    model.eval()
    predictions, actual_values = [], []
    with torch.no_grad():
        for batch_data, batch_targets, _ in test_loader:
            batch_data = batch_data.to(device)
            batch_predictions = model(batch_data)
            predictions.extend(batch_predictions.cpu().numpy())
            actual_values.extend(batch_targets.cpu().numpy())
    
    predictions, actual_values = np.array(predictions), np.array(actual_values)
    
    # Calculate regression and classification metrics.
    mse = mean_squared_error(actual_values, predictions)
    r2 = r2_score(actual_values, predictions)
    predicted_binary = (predictions <= np.log10(1000)).astype(int)
    actual_binary = (actual_values <= np.log10(1000)).astype(int)
    accuracy = accuracy_score(actual_binary, predicted_binary)
    auc = roc_auc_score(actual_binary, -predictions) # Use negative predictions as scores for AUC.
    
    print(f"📈 MODEL PERFORMANCE:")
    print(f"   - Mean Squared Error (MSE): {mse:.4f}")
    print(f"   - R-squared (R²): {r2:.4f}")
    print(f"   - Classification Accuracy: {accuracy:.4f}")
    print(f"   - Area Under ROC Curve (AUC): {auc:.4f}")
    
    # --- Visualization ---
    plt.figure(figsize=(18, 12))
    plt.suptitle("EGFR GNN Model Performance Analysis", fontsize=16, fontweight='bold')
    
    # Plot 1: Training and Validation Loss
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Predicted vs. Actual Values
    plt.subplot(2, 3, 2)
    plt.scatter(actual_values, predictions, alpha=0.6)
    plt.plot([actual_values.min(), actual_values.max()], [actual_values.min(), actual_values.max()], 'r--', lw=2)
    plt.title(f'Predicted vs. Actual log(IC50)\nR² = {r2:.3f}')
    plt.xlabel('Actual log(IC50)')
    plt.ylabel('Predicted log(IC50)')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    plt.subplot(2, 3, 3)
    residuals = actual_values - predictions
    plt.scatter(predictions, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted log(IC50)')
    plt.ylabel('Residuals')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Confusion Matrix
    plt.subplot(2, 3, 4)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(actual_binary, predicted_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Inactive', 'Active'], yticklabels=['Inactive', 'Active'])
    plt.title(f'Confusion Matrix\nAccuracy = {accuracy:.3f}')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    
    # Plot 5: Performance Metrics Summary
    plt.subplot(2, 3, 5)
    metrics = ['R²', 'Accuracy', 'AUC']
    values = [r2, accuracy, auc]
    bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon'])
    plt.title('Performance Summary')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 6: Final Summary Text
    plt.subplot(2, 3, 6)
    plt.axis('off')
    summary_text = (
        f"Final Model Performance:\n\n"
        f"  - R-squared: {r2:.3f}\n"
        f"  - Accuracy: {accuracy:.3f}\n"
        f"  - AUC: {auc:.3f}\n\n"
        f"Dataset Info:\n"
        f"  - Training Size: {len(train_graphs)}\n"
        f"  - Validation Size: {len(val_graphs)}\n"
        f"  - Test Size: {len(test_graphs)}\n\n"
        f"Model saved to 'egfr_model.pth'"
    )
    plt.text(0.05, 0.95, summary_text, ha='left', va='top', fontsize=12, family='monospace')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('gnn_training_results.png', dpi=300)
    plt.show()
    
    print("\n🎉 GNN TRAINING AND EVALUATION COMPLETE!")
    print("💾 Results saved:")
    print("   - Model: egfr_model.pth")
    print("   - Plots: gnn_training_results.png")

if __name__ == "__main__":
    main_large_dataset_training()
