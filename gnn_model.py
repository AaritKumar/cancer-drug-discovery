import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import BatchNorm, LayerNorm
import warnings
warnings.filterwarnings('ignore')

# Import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    print("✅ RDKit imported successfully")
except ImportError:
    print("❌ RDKit not found")
    exit()

class EnhancedMolecularProcessor:
    """Enhanced molecular processor for large datasets"""
    
    def __init__(self):
        # Comprehensive atom features
        self.atom_features = {
            'atomic_num': list(range(1, 120)),
            'degree': [0, 1, 2, 3, 4, 5, 6],
            'formal_charge': [-3, -2, -1, 0, 1, 2, 3],
            'chiral_tag': [0, 1, 2, 3],
            'num_Hs': [0, 1, 2, 3, 4],
            'hybridization': [0, 1, 2, 3, 4, 5, 6, 7]
        }
        
        # Precompute electronegativity values
        self.electronegativity = {
            1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 15: 2.19, 16: 2.58, 
            17: 3.16, 35: 2.96, 53: 2.66, 5: 2.04, 14: 1.90, 33: 2.18, 34: 2.55
        }
        
    def atom_to_features(self, atom):
        """Convert atom to comprehensive feature vector"""
        features = []
        
        # Basic atomic properties
        atomic_num = atom.GetAtomicNum()
        
        # One-hot encoding for common elements (1-20)
        features.extend([1 if atomic_num == x else 0 for x in range(1, 21)])
        
        # Normalized atomic number for less common elements
        features.append(atomic_num / 100.0)
        
        # Connectivity features
        degree = atom.GetDegree()
        features.extend([1 if degree == x else 0 for x in range(7)])
        features.append(degree / 6.0)  # Normalized degree
        
        # Chemical properties
        features.append(atom.GetFormalCharge())
        features.append(atom.GetIsAromatic())
        features.append(atom.IsInRing())
        
        # Hybridization
        try:
            hyb = int(atom.GetHybridization())
            features.extend([1 if hyb == x else 0 for x in range(8)])
        except:
            features.extend([0] * 8)
        
        # Hydrogen count
        features.append(atom.GetTotalNumHs())
        
        # Fixed: Use GetValence(getExplicit=False) instead of GetImplicitValence()
        try:
            features.append(atom.GetValence(getExplicit=False))
        except:
            features.append(0)
        
        # Electronegativity
        features.append(self.electronegativity.get(atomic_num, 2.0))
        
        # Ring information
        features.append(atom.IsInRingSize(3))
        features.append(atom.IsInRingSize(4))
        features.append(atom.IsInRingSize(5))
        features.append(atom.IsInRingSize(6))
        features.append(atom.IsInRingSize(7))
        
        # Additional chemical features
        features.append(1 if atomic_num in [6, 7, 8, 16] else 0)  # Common biological atoms
        features.append(1 if atomic_num in [9, 17, 35, 53] else 0)  # Halogens
        
        return np.array(features, dtype=np.float32)
    
    def bond_to_features(self, bond):
        """Extract bond features"""
        features = []
        
        # Bond type
        bond_type = bond.GetBondTypeAsDouble()
        features.extend([1 if bond_type == x else 0 for x in [1.0, 1.5, 2.0, 3.0]])
        
        # Bond properties
        features.append(bond.GetIsAromatic())
        features.append(bond.GetIsConjugated())
        features.append(bond.IsInRing())
        
        # Stereo information
        try:
            stereo = int(bond.GetStereo())
            features.extend([1 if stereo == x else 0 for x in range(6)])
        except:
            features.extend([0] * 6)
        
        return np.array(features, dtype=np.float32)
    
    def smiles_to_graph(self, smiles):
        """Convert SMILES to enhanced molecular graph"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Add hydrogens for complete representation
            mol = Chem.AddHs(mol)
            
            # Get atom features
            atom_features = []
            for atom in mol.GetAtoms():
                atom_features.append(self.atom_to_features(atom))
            
            x = torch.tensor(atom_features, dtype=torch.float)
            
            # Get edge indices and features
            edge_indices = []
            edge_features = []
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                bond_feat = self.bond_to_features(bond)
                
                edge_indices.extend([[i, j], [j, i]])
                edge_features.extend([bond_feat, bond_feat])
            
            if len(edge_indices) > 0:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_features, dtype=torch.float)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, len(edge_features[0]) if edge_features else 10), dtype=torch.float)
            
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return None

class AdvancedEGFRNet(nn.Module):
    """Advanced Graph Neural Network for large-scale EGFR prediction"""
    
    def __init__(self, num_node_features, num_edge_features, hidden_dim=256, num_layers=4, 
                 num_heads=8, dropout=0.1, use_batch_norm=True):
        super(AdvancedEGFRNet, self).__init__()
        
        # Input projections
        self.node_embedding = nn.Linear(num_node_features, hidden_dim)
        self.edge_embedding = nn.Linear(num_edge_features, hidden_dim)
        
        # Graph attention layers with residual connections
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                           dropout=dropout, edge_dim=hidden_dim, concat=True)
                )
            else:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                           dropout=dropout, edge_dim=hidden_dim, concat=True)
                )
            
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_dim))
            else:
                self.batch_norms.append(nn.Identity())
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Global pooling combination
        self.pool_dim = hidden_dim * 3  # mean + max + add pooling
        
        # Enhanced prediction head
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
            
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for better training"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Input embeddings
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr) if edge_attr.size(0) > 0 else None
        
        # Graph attention layers with residual connections
        for i, (gat, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            residual = x
            
            if edge_attr is not None:
                x = gat(x, edge_index, edge_attr)
            else:
                x = gat(x, edge_index)
            
            x = bn(x) if x.size(0) > 1 else x
            x = F.relu(x)
            x = self.dropout(x)
            
            # Residual connection (if dimensions match)
            if residual.shape == x.shape:
                x = x + residual
        
        # Multiple global pooling strategies
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_add = global_add_pool(x, batch)
        
        # Concatenate pooling results
        x = torch.cat([x_mean, x_max, x_add], dim=1)
        
        # Final prediction
        x = self.prediction_head(x)
        
        return x.squeeze()

class LargeDatasetLoader:
    """Efficient data loader for large datasets"""
    
    def __init__(self, csv_file, processor, test_size=0.2, val_size=0.1, random_state=42):
        self.csv_file = csv_file
        self.processor = processor
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
    
    def load_and_process_data(self):
        """Load and process large dataset efficiently"""
        
        print("📊 LOADING LARGE DATASET")
        print("=" * 30)
        
        # Load dataset
        df = pd.read_csv(self.csv_file)
        print(f"✅ Loaded {len(df)} compounds")
        
        # Stratified sampling for balanced training
        if len(df) > 50000:
            print("🎯 Large dataset detected - using stratified sampling...")
            df_balanced = self._stratified_sample(df, max_size=20000)
            print(f"📊 Sampled {len(df_balanced)} compounds for training")
        else:
            df_balanced = df.copy()
        
        # Convert to graphs
        print("🧬 Converting molecules to graphs...")
        graphs, activities, names = self._molecules_to_graphs(df_balanced)
        
        # Create labels
        binary_labels = torch.tensor([1 if act <= np.log10(1000) else 0 for act in activities], dtype=torch.float)
        
        # Split data
        train_val_graphs, test_graphs, train_val_activities, test_activities, \
        train_val_binary, test_binary, train_val_names, test_names = train_test_split(
            graphs, activities, binary_labels, names, 
            test_size=self.test_size, random_state=self.random_state, stratify=binary_labels
        )
        
        # Further split training into train/validation
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
    
    def _stratified_sample(self, df, max_size=20000):
        """Create balanced sample from large dataset"""
        
        # Create activity bins for stratified sampling
        df['activity_bin'] = pd.cut(df['log_activity'], bins=10, labels=False)
        
        # Sample proportionally from each bin
        samples_per_bin = max_size // 10
        sampled_dfs = []
        
        for bin_val in range(10):
            bin_df = df[df['activity_bin'] == bin_val]
            if len(bin_df) > samples_per_bin:
                bin_sample = bin_df.sample(n=samples_per_bin, random_state=self.random_state)
            else:
                bin_sample = bin_df.copy()
            sampled_dfs.append(bin_sample)
        
        return pd.concat(sampled_dfs, ignore_index=True)
    
    def _molecules_to_graphs(self, df):
        """Convert molecules to graphs with progress tracking"""
        
        graphs = []
        activities = []
        names = []
        
        total = len(df)
        failed_count = 0
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"   Processing {idx+1}/{total} ({(idx+1)/total*100:.1f}%)")
            
            smiles = row['canonical_smiles']
            activity = row['log_activity'] if 'log_activity' in row else np.log10(row['activity_value_nm'])
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
        
        activities = torch.tensor(activities, dtype=torch.float)
        
        return graphs, activities, names

class EnhancedTrainer:
    """Enhanced trainer for large-scale GNN training"""
    
    def __init__(self, model, train_loader, val_loader, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Enhanced optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.001, weight_decay=1e-4, betas=(0.9, 0.999)
        )
        # Fixed: Removed verbose=True parameter
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.criterion = nn.MSELoss()
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 20
    
    def enhanced_collate(self, batch):
        """Enhanced collate function for large batches"""
        graphs, targets, names = zip(*batch)
        
        # Combine graphs into batch
        batch_data = Data()
        
        x_list = [graph.x for graph in graphs]
        batch_data.x = torch.cat(x_list, dim=0)
        
        edge_index_list = []
        edge_attr_list = []
        edge_offset = 0
        batch_list = []
        
        for i, graph in enumerate(graphs):
            edge_index = graph.edge_index + edge_offset
            edge_index_list.append(edge_index)
            
            if hasattr(graph, 'edge_attr') and graph.edge_attr.size(0) > 0:
                edge_attr_list.append(graph.edge_attr)
            
            batch_vector = torch.full((graph.x.size(0),), i, dtype=torch.long)
            batch_list.append(batch_vector)
            
            edge_offset += graph.x.size(0)
        
        if edge_index_list:
            batch_data.edge_index = torch.cat(edge_index_list, dim=1)
        else:
            batch_data.edge_index = torch.empty((2, 0), dtype=torch.long)
        
        if edge_attr_list:
            batch_data.edge_attr = torch.cat(edge_attr_list, dim=0)
        else:
            batch_data.edge_attr = torch.empty((0, 10), dtype=torch.float)
        
        batch_data.batch = torch.cat(batch_list, dim=0)
        
        targets_tensor = torch.stack(list(targets))
        
        return batch_data.to(self.device), targets_tensor.to(self.device), names
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_data, batch_targets, batch_names in self.train_loader:
            self.optimizer.zero_grad()
            
            predictions = self.model(batch_data)
            loss = self.criterion(predictions, batch_targets)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data, batch_targets, batch_names in self.val_loader:
                predictions = self.model(batch_data)
                loss = self.criterion(predictions, batch_targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def step_scheduler(self, val_loss):
        """Step scheduler with manual verbose logging"""
        old_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step(val_loss)
        new_lr = self.optimizer.param_groups[0]['lr']
        
        if old_lr != new_lr:
            print(f"📉 Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
    
    def train(self, num_epochs=100):
        """Full training loop with early stopping"""
        
        print(f"🚀 ENHANCED TRAINING STARTED")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling with manual verbose logging
            self.step_scheduler(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_large_egfr_model.pth')
            else:
                self.patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            if self.patience_counter >= self.max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_large_egfr_model.pth'))
        print("✅ Training completed!")
        
        return self.train_losses, self.val_losses

def main_large_dataset_training():
    """Main function for large dataset training"""
    
    print("🧠 ENHANCED EGFR DRUG DISCOVERY - LARGE SCALE")
    print("=" * 55)
    
    # Check for large dataset
    import glob
    large_files = glob.glob('egfr_large_dataset_*.csv')
    
    if not large_files:
        print("❌ No large dataset found!")
        print("Please run large_dataset_download.py first")
        return
    
    # Use the largest dataset
    dataset_file = max(large_files, key=lambda x: int(x.split('_')[3]))
    print(f"📊 Using dataset: {dataset_file}")
    
    # Initialize processor and data loader
    processor = EnhancedMolecularProcessor()
    data_loader = LargeDatasetLoader(dataset_file, processor)
    
    # Load and process data
    (train_graphs, train_activities, train_binary, train_names,
     val_graphs, val_activities, val_binary, val_names,
     test_graphs, test_activities, test_binary, test_names) = data_loader.load_and_process_data()
    
    # Create datasets and data loaders
    from torch.utils.data import Dataset, DataLoader as TorchDataLoader
    
    class MolDataset(Dataset):
        def __init__(self, graphs, targets, names):
            self.graphs = graphs
            self.targets = targets
            self.names = names
        
        def __len__(self):
            return len(self.graphs)
        
        def __getitem__(self, idx):
            return self.graphs[idx], self.targets[idx], self.names[idx]
    
    # Create datasets
    train_dataset = MolDataset(train_graphs, train_activities, train_names)
    val_dataset = MolDataset(val_graphs, val_activities, val_names)
    test_dataset = MolDataset(test_graphs, test_activities, test_names)
    
    # Initialize model
    sample_graph = train_graphs[0]
    num_node_features = sample_graph.x.shape[1]
    num_edge_features = sample_graph.edge_attr.shape[1] if hasattr(sample_graph, 'edge_attr') else 10
    
    model = AdvancedEGFRNet(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        dropout=0.1
    )
    
    print(f"🧠 Model architecture:")
    print(f"   Node features: {num_node_features}")
    print(f"   Edge features: {num_edge_features}")
    print(f"   Hidden dimension: 256")
    print(f"   Layers: 4")
    print(f"   Attention heads: 8")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data loaders
    train_loader = TorchDataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = TorchDataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    trainer = EnhancedTrainer(model, train_loader, val_loader, device)
    
    # Set collate function
    train_loader.collate_fn = trainer.enhanced_collate
    val_loader.collate_fn = trainer.enhanced_collate
    test_loader.collate_fn = trainer.enhanced_collate
    
    # Train model
    train_losses, val_losses = trainer.train(num_epochs=100)
    
    # Evaluate final model
    print("\n📊 FINAL MODEL EVALUATION")
    print("=" * 30)
    
    model.eval()
    predictions = []
    actual_values = []
    actual_binary = []
    
    with torch.no_grad():
        for batch_data, batch_targets, batch_names in test_loader:
            batch_predictions = model(batch_data)
            predictions.extend(batch_predictions.cpu().numpy())
            actual_values.extend(batch_targets.cpu().numpy())
    
    predictions = np.array(predictions)
    actual_values = np.array(actual_values)
    
    # Calculate metrics
    mse = mean_squared_error(actual_values, predictions)
    r2 = r2_score(actual_values, predictions)
    
    # Binary classification metrics
    predicted_binary = (predictions <= np.log10(1000)).astype(int)
    actual_binary = (actual_values <= np.log10(1000)).astype(int)
    accuracy = accuracy_score(actual_binary, predicted_binary)
    
    try:
        auc = roc_auc_score(actual_binary, -predictions)
    except:
        auc = 0.5
    
    print(f"📈 ENHANCED MODEL PERFORMANCE:")
    print(f"   MSE: {mse:.4f}")
    print(f"   R²: {r2:.4f}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   AUC: {auc:.4f}")
    
    # Create comprehensive results
    results_df = pd.DataFrame({
        'Actual_IC50': 10**actual_values,
        'Predicted_IC50': 10**predictions,
        'Actual_Class': ['Active' if x == 1 else 'Inactive' for x in actual_binary],
        'Predicted_Class': ['Active' if x == 1 else 'Inactive' for x in predicted_binary]
    })
    
    results_df.to_csv('large_dataset_results.csv', index=False)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Training curves
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Actual vs Predicted
    plt.subplot(2, 3, 2)
    plt.scatter(actual_values, predictions, alpha=0.6)
    plt.plot([actual_values.min(), actual_values.max()], 
             [actual_values.min(), actual_values.max()], 'r--', lw=2)
    plt.xlabel('Actual log(IC50)')
    plt.ylabel('Predicted log(IC50)')
    plt.title(f'Actual vs Predicted\nR² = {r2:.3f}')
    plt.grid(True, alpha=0.3)
    
    # Residuals
    plt.subplot(2, 3, 3)
    residuals = actual_values - predictions
    plt.scatter(predictions, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted log(IC50)')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    # Activity distribution
    plt.subplot(2, 3, 4)
    plt.hist(10**actual_values, bins=30, alpha=0.6, label='Actual', color='blue')
    plt.hist(10**predictions, bins=30, alpha=0.6, label='Predicted', color='red')
    plt.xlabel('IC50 (nM)')
    plt.ylabel('Count')
    plt.title('Activity Distribution')
    plt.legend()
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    # Confusion matrix
    plt.subplot(2, 3, 5)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(actual_binary, predicted_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Inactive', 'Active'],
                yticklabels=['Inactive', 'Active'])
    plt.title(f'Confusion Matrix\nAccuracy = {accuracy:.3f}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Performance summary
    plt.subplot(2, 3, 6)
    metrics = ['R²', 'Accuracy', 'AUC']
    values = [r2, accuracy, auc]
    colors = ['green' if v > 0.8 else 'orange' if v > 0.7 else 'red' for v in values]
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.ylabel('Score')
    plt.title('Model Performance Summary')
    plt.ylim(0, 1)
    
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('large_dataset_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎉 LARGE-SCALE TRAINING COMPLETE!")
    print(f"💾 Results saved:")
    print(f"   Model: best_large_egfr_model.pth")
    print(f"   Results: large_dataset_results.csv")
    print(f"   Plots: large_dataset_training_results.png")
    
    # Performance assessment
    if r2 > 0.8 and accuracy > 0.85:
        print(f"\n🏆 EXCELLENT PERFORMANCE!")
        print(f"Your enhanced model is ready for high-confidence virtual screening!")
    elif r2 > 0.7 and accuracy > 0.8:
        print(f"\n✅ GOOD PERFORMANCE!")
        print(f"Your model shows significant improvement and is suitable for virtual screening!")
    else:
        print(f"\n📈 IMPROVED PERFORMANCE!")
        print(f"Your model shows improvement over the small dataset version!")

if __name__ == "__main__":
    main_large_dataset_training()