#!/usr/bin/env python3
"""
EGFR Cancer Drug Discovery Pipeline
Uses your trained EGFR GNN model to screen ChEMBL compounds for potential new cancer drugs

Usage: python drug_discovery.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
import os
import pickle
from tqdm import tqdm
import warnings

# Import the model architecture and processor from your training script
from gnn_model import AdvancedEGFRNet, EnhancedMolecularProcessor

warnings.filterwarnings('ignore')

class EGFRDrugDiscovery:
    def __init__(self, model_path="egfr_model.pth", dataset_path="drug_discovery_dataset/chembl_drug_discovery_dataset.csv"):
        """
        Initialize the drug discovery pipeline
        
        Args:
            model_path: Path to your trained EGFR GNN model
            dataset_path: Path to the processed ChEMBL dataset
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = EnhancedMolecularProcessor()
        
        # Results storage
        self.discoveries = []
        
        print(f"🔬 EGFR Cancer Drug Discovery Pipeline")
        print(f"💻 Device: {self.device}")
        print(f"📁 Model: {model_path}")
        print(f"📊 Dataset: {dataset_path}")
    
    def load_model(self):
        """Load your trained EGFR GNN model"""
        print(f"\n🧠 Loading EGFR model...")
        
        if not os.path.exists(self.model_path):
            print(f"❌ Model file not found: {self.model_path}")
            print("💡 Make sure your trained EGFR model is available.")
            return None
        
        try:
            # Determine model feature dimensions from a sample molecule graph
            sample_graph = self.smiles_to_graph("CCO")  # Use ethanol as a dummy molecule
            if sample_graph is None:
                print("❌ Could not create a sample graph to determine model dimensions.")
                return None
            num_node_features = sample_graph.x.shape[1]
            num_edge_features = sample_graph.edge_attr.shape[1]

            # Instantiate the model with the correct dimensions
            model = AdvancedEGFRNet(
                num_node_features=num_node_features,
                num_edge_features=num_edge_features
            )
            
            # Load the saved state dictionary
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            print(f"✅ Model loaded successfully on {self.device}")
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Make sure the model file is compatible and the model architecture is correct.")
            return None
    
    def load_chembl_dataset(self):
        """Load the processed ChEMBL dataset"""
        print(f"\n📥 Loading ChEMBL drug discovery dataset...")
        
        if not os.path.exists(self.dataset_path):
            print(f"❌ Dataset not found: {self.dataset_path}")
            print("💡 Make sure the discovery dataset is available in the 'drug_discovery_dataset' folder.")
            return None
        
        try:
            df = pd.read_csv(self.dataset_path)
            print(f"✅ Loaded {len(df):,} compounds")
            
            # Validate required columns
            required_cols = ['SMILES']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"❌ Missing required columns: {missing_cols}")
                return None
            
            # Remove invalid SMILES
            df = df.dropna(subset=['SMILES'])
            df = df[df['SMILES'].str.len() > 5]
            
            print(f"📊 Valid compounds for screening: {len(df):,}")
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def smiles_to_graph(self, smiles):
        """
        Convert SMILES to molecular graph for GNN using the processor from the training script.
        """
        return self.processor.smiles_to_graph(smiles)
    
    def predict_egfr_activity(self, model, graph_data):
        """
        Predict EGFR activity for a molecular graph.
        The model is a regressor for log_IC50, so we convert the output
        to a pseudo-probability for screening and ranking.
        """
        try:
            with torch.no_grad():
                graph_data = graph_data.to(self.device)
                
                # Predict the log_IC50 value
                prediction = model(graph_data)
                
                # Convert log_IC50 to a pseudo-probability. A lower log_IC50 is better.
                # We use a sigmoid function centered around a threshold (e.g., log10(1000) = 3).
                center = 3.0  # Corresponds to 1000 nM
                width = 2.0   # Controls the steepness of the probability curve
                probability = torch.sigmoid(-(prediction - center) / width).item()
                
                return probability
                
        except Exception as e:
            return None
    
    def calculate_drug_properties(self, smiles):
        """Calculate additional drug properties for discovered compounds"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            properties = {
                'Molecular_Weight': Descriptors.MolWt(mol),
                'LogP': Crippen.MolLogP(mol),
                'H_Bond_Donors': Lipinski.NumHDonors(mol),
                'H_Bond_Acceptors': Lipinski.NumHAcceptors(mol),
                'TPSA': Descriptors.TPSA(mol),
                'Rotatable_Bonds': Descriptors.NumRotatableBonds(mol),
                'Aromatic_Rings': Descriptors.NumAromaticRings(mol),
                'Lipinski_Violations': sum(1 for rule in Lipinski.rdMolDescriptors.GetLipinskiRuleViolations(mol) if rule[1]),
                'QED_Score': Descriptors.qed(mol),  # Drug-likeness score
            }
            
            return properties
            
        except Exception as e:
            return {}
    
    def screen_compounds(self, model, df, batch_size=100, min_probability=0.7):
        """Screen compounds for EGFR activity"""
        print(f"\n🔬 Screening {len(df):,} compounds for EGFR activity...")
        print(f"🎯 Minimum probability threshold: {min_probability}")
        
        discoveries = []
        failed_count = 0
        
        # Process in batches
        for i in tqdm(range(0, len(df), batch_size), desc="Screening batches"):
            batch_df = df.iloc[i:i+batch_size]
            
            for idx, row in batch_df.iterrows():
                smiles = row['SMILES']
                
                # Convert to graph
                graph_data = self.smiles_to_graph(smiles)
                if graph_data is None:
                    failed_count += 1
                    continue
                
                # Predict EGFR activity
                probability = self.predict_egfr_activity(model, graph_data)
                if probability is None:
                    failed_count += 1
                    continue
                
                # Check if compound shows promising activity
                if probability >= min_probability:
                    # Calculate additional properties
                    properties = self.calculate_drug_properties(smiles)
                    
                    discovery = {
                        'ChEMBL_ID': row.get('ChEMBL_ID', f'Discovery_{i+idx}'),
                        'SMILES': smiles,
                        'EGFR_Probability': probability,
                        'Confidence_Level': self.get_confidence_level(probability),
                        **properties
                    }
                    
                    discoveries.append(discovery)
        
        print(f"✅ Screening complete!")
        print(f"🎯 Promising compounds found: {len(discoveries)}")
        print(f"⚠️  Failed predictions: {failed_count}")
        
        return discoveries
    
    def get_confidence_level(self, probability):
        """Assign confidence level based on probability"""
        if probability >= 0.9:
            return "Very High"
        elif probability >= 0.8:
            return "High"
        elif probability >= 0.7:
            return "Moderate"
        else:
            return "Low"
    
    def rank_discoveries(self, discoveries):
        """Rank discoveries by multiple criteria"""
        print(f"\n📊 Ranking {len(discoveries)} discoveries...")
        
        df = pd.DataFrame(discoveries)
        
        # Create composite score
        df['Drug_Likeness_Score'] = 0
        
        # EGFR probability (40% weight)
        df['Drug_Likeness_Score'] += df['EGFR_Probability'] * 0.4
        
        # QED score if available (20% weight)
        if 'QED_Score' in df.columns:
            qed_normalized = df['QED_Score'].fillna(0)
            df['Drug_Likeness_Score'] += qed_normalized * 0.2
        
        # Lipinski compliance (20% weight)
        if 'Lipinski_Violations' in df.columns:
            lipinski_score = (2 - df['Lipinski_Violations'].clip(0, 2)) / 2
            df['Drug_Likeness_Score'] += lipinski_score * 0.2
        
        # Molecular weight preference (10% weight) - prefer 300-450 Da
        if 'Molecular_Weight' in df.columns:
            mw_score = 1 - abs(df['Molecular_Weight'] - 375) / 375
            mw_score = mw_score.clip(0, 1)
            df['Drug_Likeness_Score'] += mw_score.fillna(0) * 0.1
        
        # LogP preference (10% weight) - prefer 1-3
        if 'LogP' in df.columns:
            logp_score = 1 - abs(df['LogP'] - 2) / 5
            logp_score = logp_score.clip(0, 1)
            df['Drug_Likeness_Score'] += logp_score.fillna(0) * 0.1
        
        # Sort by composite score
        df = df.sort_values('Drug_Likeness_Score', ascending=False)
        
        return df
    
    def save_discoveries(self, discoveries_df, filename="egfr_drug_discoveries.csv"):
        """Save discoveries to file"""
        output_path = filename
        discoveries_df.to_csv(output_path, index=False)
        print(f"💾 Discoveries saved to: {output_path}")
        return output_path
    
    def save_top_discoveries_text(self, discoveries_df, top_n=100, filename="top_100_discoveries.txt"):
        """Save a detailed summary of top discoveries to a text file"""
        print(f"\n💾 Saving top {top_n} discoveries to {filename}...")
        
        with open(filename, 'w') as f:
            f.write("🏆 TOP EGFR DRUG DISCOVERIES 🏆\n")
            f.write("=" * 80 + "\n\n")
            
            for i, (_, row) in enumerate(discoveries_df.head(top_n).iterrows()):
                f.write(f"🥇 RANK #{i+1}\n")
                f.write("-" * 30 + "\n")
                f.write(f"  ChEMBL ID: {row['ChEMBL_ID']}\n")
                f.write(f"  SMILES: {row['SMILES']}\n")
                f.write(f"  EGFR Probability: {row['EGFR_Probability']:.4f} ({row['Confidence_Level']} confidence)\n")
                f.write(f"  Drug-Likeness Score: {row['Drug_Likeness_Score']:.4f}\n\n")
                
                f.write("  Molecular Properties:\n")
                if 'Molecular_Weight' in row and pd.notna(row['Molecular_Weight']):
                    f.write(f"    - Molecular Weight: {row['Molecular_Weight']:.2f} Da\n")
                if 'LogP' in row and pd.notna(row['LogP']):
                    f.write(f"    - LogP: {row['LogP']:.2f}\n")
                if 'H_Bond_Donors' in row and pd.notna(row['H_Bond_Donors']):
                    f.write(f"    - H-bond Donors: {int(row['H_Bond_Donors'])}\n")
                if 'H_Bond_Acceptors' in row and pd.notna(row['H_Bond_Acceptors']):
                    f.write(f"    - H-bond Acceptors: {int(row['H_Bond_Acceptors'])}\n")
                if 'TPSA' in row and pd.notna(row['TPSA']):
                    f.write(f"    - TPSA: {row['TPSA']:.2f}\n")
                if 'Rotatable_Bonds' in row and pd.notna(row['Rotatable_Bonds']):
                    f.write(f"    - Rotatable Bonds: {int(row['Rotatable_Bonds'])}\n")
                if 'Aromatic_Rings' in row and pd.notna(row['Aromatic_Rings']):
                    f.write(f"    - Aromatic Rings: {int(row['Aromatic_Rings'])}\n")
                if 'QED_Score' in row and pd.notna(row['QED_Score']):
                    f.write(f"    - QED Drug-likeness: {row['QED_Score']:.3f}\n")
                if 'Lipinski_Violations' in row and pd.notna(row['Lipinski_Violations']):
                    f.write(f"    - Lipinski Violations: {int(row['Lipinski_Violations'])}\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
        
        print(f"✅ Top discoveries text file saved.")
    
    def print_top_discoveries(self, discoveries_df, top_n=20):
        """Print summary of top discoveries"""
        print(f"\n🏆 TOP {min(top_n, len(discoveries_df))} EGFR DRUG DISCOVERIES")
        print("=" * 80)
        
        for i, (_, row) in enumerate(discoveries_df.head(top_n).iterrows()):
            print(f"\n🥇 RANK #{i+1}")
            print(f"   ChEMBL ID: {row['ChEMBL_ID']}")
            print(f"   SMILES: {row['SMILES']}")
            print(f"   EGFR Probability: {row['EGFR_Probability']:.3f} ({row['Confidence_Level']} confidence)")
            print(f"   Drug-Likeness Score: {row['Drug_Likeness_Score']:.3f}")
            
            # Print molecular properties
            print(f"   Properties:")
            if 'Molecular_Weight' in row and pd.notna(row['Molecular_Weight']):
                print(f"     • Molecular Weight: {row['Molecular_Weight']:.1f} Da")
            if 'LogP' in row and pd.notna(row['LogP']):
                print(f"     • LogP: {row['LogP']:.2f}")
            if 'H_Bond_Donors' in row and pd.notna(row['H_Bond_Donors']):
                print(f"     • H-bond Donors: {int(row['H_Bond_Donors'])}")
            if 'H_Bond_Acceptors' in row and pd.notna(row['H_Bond_Acceptors']):
                print(f"     • H-bond Acceptors: {int(row['H_Bond_Acceptors'])}")
            if 'QED_Score' in row and pd.notna(row['QED_Score']):
                print(f"     • QED Drug-likeness: {row['QED_Score']:.3f}")
            if 'Lipinski_Violations' in row and pd.notna(row['Lipinski_Violations']):
                print(f"     • Lipinski Violations: {int(row['Lipinski_Violations'])}")
    
    def run_discovery_pipeline(self, min_probability=0.7, max_compounds=50000):
        """Run the complete drug discovery pipeline"""
        print(f"\n🚀 STARTING EGFR CANCER DRUG DISCOVERY")
        print("=" * 60)
        
        # Load model
        model = self.load_model()
        if model is None:
            print("❌ Cannot proceed without model")
            return None
        
        # Load dataset
        df = self.load_chembl_dataset()
        if df is None:
            print("❌ Cannot proceed without dataset")
            return None
        
        # Limit dataset size if needed
        if len(df) > max_compounds:
            print(f"📊 Limiting dataset to {max_compounds:,} compounds for faster screening")
            df = df.sample(n=max_compounds, random_state=42)
        
        # Screen compounds
        discoveries = self.screen_compounds(model, df, min_probability=min_probability)
        
        if not discoveries:
            print("\n😞 No promising compounds found. Try lowering the probability threshold or checking your model.")
            return None
        
        # Rank discoveries
        discoveries_df = self.rank_discoveries(discoveries)
        
        # Save results
        output_path = self.save_discoveries(discoveries_df)
        
        # Save top 100 to text file
        self.save_top_discoveries_text(discoveries_df, top_n=100)
        
        # Print top discoveries
        self.print_top_discoveries(discoveries_df)
        
        print(f"\n🎉 DRUG DISCOVERY COMPLETE!")
        print(f"💊 Total discoveries: {len(discoveries_df)}")
        print(f"📁 Results saved to: {output_path}")
        print(f"🔬 Ready for experimental validation!")
        
        return discoveries_df

def main():
    """Main execution function"""
    # Initialize discovery pipeline with the correct model and dataset paths
    discovery = EGFRDrugDiscovery(
        model_path="egfr_model.pth",
        dataset_path="drug_discovery_dataset/chembl_drug_discovery_dataset.csv"
    )
    
    # Run the full discovery pipeline
    results = discovery.run_discovery_pipeline(
        min_probability=0.7,  # Minimum activity probability to be considered a "hit"
        max_compounds=100000   # Limit number of compounds for a quicker run
    )
    
    if results is not None:
        print(f"\n🎯 SUCCESS! Discovered {len(results)} potential EGFR inhibitors!")
        print("🔬 Next steps:")
        print("   1. Review the top candidates in 'egfr_drug_discoveries.csv' and 'top_100_discoveries.txt'")
        print("   2. Perform molecular docking studies for the best hits")
        print("   3. Consider experimental validation for the most promising compounds")
        print("   4. Check literature for known activities or similar scaffolds")

if __name__ == "__main__":
    main()