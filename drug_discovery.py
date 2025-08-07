#!/usr/bin/env python3
"""
EGFR Cancer Drug Discovery Pipeline
===================================

This script represents the core of the drug discovery effort. It takes the trained
Graph Neural Network (GNN) model and uses it to perform a large-scale virtual screen
of a massive library of chemical compounds. The goal is to identify novel molecules
that the model predicts will be highly active against the EGFR cancer target.

The pipeline executes the following major steps:

1.  **Load the Trained Model**: It loads the `AdvancedEGFRNet` model and its saved weights
    (`egfr_model.pth`) that were trained in the `gnn_model.py` script.

2.  **Load the Discovery Dataset**: It loads the large dataset of drug-like molecules
    that was prepared by the `chembl_processor.py` script. This dataset contains
    hundreds of thousands of compounds to be screened.

3.  **Screen Compounds**: The script iterates through each compound in the discovery dataset.
    For each molecule, it:
    a. Converts the SMILES string to the same graph representation used during training.
    b. Feeds the graph into the GNN model to predict its bioactivity (log_IC50).
    c. Converts the predicted log_IC50 value into a pseudo-probability, which provides
       an intuitive score of how likely the compound is to be an active EGFR inhibitor.

4.  **Identify "Hits"**: Compounds that score above a certain probability threshold are
    considered "hits" – promising candidates for new drugs.

5.  **Calculate Properties**: For each hit, it calculates a wide range of important
    medicinal chemistry properties, such as Molecular Weight, LogP, and Quantitative
    Estimate of Drug-likeness (QED).

6.  **Rank Discoveries**: The script uses a composite scoring function to rank the hits.
    This score considers not only the predicted activity but also the overall
    drug-likeness of the molecule, ensuring that the top-ranked compounds are both
    potent and have favorable properties for further development.

7.  **Save Results**: The final ranked list of discoveries is saved to both a comprehensive
    CSV file and a detailed, human-readable text file for the top 100 candidates.

This script operationalizes the trained AI model, turning its predictive power into a
practical tool for identifying potential new cancer therapies from a vast chemical space.

Usage:
    python drug_discovery.py
"""

# --- Core Imports ---
import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
import warnings

# --- PyTorch and RDKit Imports ---
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski

# --- Custom Module Imports ---
# Import the model architecture (AdvancedEGFRNet) and the molecule-to-graph processor
# from the `gnn_model.py` script. This is crucial to ensure that the data processing
# and model architecture are identical to how the model was trained.
from gnn_model import AdvancedEGFRNet, EnhancedMolecularProcessor

# Suppress minor warnings for a cleaner console output.
warnings.filterwarnings('ignore')

# =====================================================================================
#  The Main Drug Discovery Class
# =====================================================================================

class EGFRDrugDiscovery:
    """
    A class to encapsulate the entire drug discovery pipeline, from loading the model
    and data to screening compounds and saving the results.
    """
    
    def __init__(self, model_path="egfr_model.pth", dataset_path="drug_discovery_dataset/chembl_drug_discovery_dataset.csv"):
        """
        Initializes the drug discovery pipeline.

        Args:
            model_path (str): The file path to the trained GNN model weights.
            dataset_path (str): The file path to the dataset of compounds to be screened.
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        # Automatically select the GPU for computation if available, otherwise default to the CPU.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Instantiate the same molecular processor used during training.
        self.processor = EnhancedMolecularProcessor()
        
        print(f"🔬 EGFR Cancer Drug Discovery Pipeline")
        print(f"💻 Device: {self.device}")
        print(f"📁 Model: {model_path}")
        print(f"📊 Dataset: {dataset_path}")
    
    def load_model(self):
        """
        Loads the pre-trained EGFR GNN model from the specified path.
        It dynamically determines the required input feature dimensions for the model.
        """
        print(f"\n🧠 Loading EGFR model...")
        
        if not os.path.exists(self.model_path):
            print(f"❌ Model file not found: {self.model_path}")
            return None
        
        try:
            # To initialize the model, we need to know the number of node and edge features it was trained on.
            # We can determine this by creating a dummy graph and inspecting its feature dimensions.
            sample_graph = self.smiles_to_graph("CCO")  # "CCO" is the SMILES for ethanol, a simple molecule.
            if sample_graph is None:
                print("❌ Could not create a sample graph to determine model dimensions.")
                return None
            num_node_features = sample_graph.x.shape[1]
            num_edge_features = sample_graph.edge_attr.shape[1]

            # Now we can instantiate the model with the correct dimensions.
            model = AdvancedEGFRNet(
                num_node_features=num_node_features,
                num_edge_features=num_edge_features
            )
            
            # Load the saved weights (the "state dictionary") into the model architecture.
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.to(self.device) # Move the model to the selected device (GPU or CPU).
            model.eval() # Set the model to evaluation mode, which disables layers like dropout.
            print(f"✅ Model loaded successfully on {self.device}")
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def load_chembl_dataset(self):
        """
        Loads the prepared ChEMBL dataset of compounds to be screened.
        """
        print(f"\n📥 Loading ChEMBL drug discovery dataset...")
        
        if not os.path.exists(self.dataset_path):
            print(f"❌ Dataset not found: {self.dataset_path}")
            return None
        
        try:
            df = pd.read_csv(self.dataset_path)
            print(f"✅ Loaded {len(df):,} compounds")
            
            # Ensure the dataset has the essential 'SMILES' column.
            if 'SMILES' not in df.columns:
                print(f"❌ Missing required column: 'SMILES'")
                return None
            
            # Perform basic cleaning to remove invalid entries.
            df = df.dropna(subset=['SMILES'])
            df = df[df['SMILES'].str.len() > 5]
            
            print(f"📊 Valid compounds for screening: {len(df):,}")
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def smiles_to_graph(self, smiles: str):
        """
        Converts a SMILES string to a molecular graph using the processor from the training script.
        """
        return self.processor.smiles_to_graph(smiles)
    
    def predict_egfr_activity(self, model, graph_data):
        """
        Predicts the EGFR activity for a single molecular graph. Since the model was trained
        as a regressor to predict log_IC50, this function converts the regression output into
        an intuitive pseudo-probability score for ranking.
        """
        try:
            with torch.no_grad(): # Disable gradient calculations to speed up inference.
                graph_data = graph_data.to(self.device)
                
                # Get the model's raw prediction (predicted log_IC50).
                prediction = model(graph_data)
                
                # --- Conversion from Regression to Pseudo-Probability ---
                # A lower log_IC50 value means higher potency. We want to transform this so that
                # a higher final score is better. We use a sigmoid (logistic) function for this.
                # The sigmoid function squashes any real number into a (0, 1) range, like a probability.
                center = 3.0  # Corresponds to a log_IC50 of 3, which is an IC50 of 1000 nM (a common activity cutoff).
                width = 2.0   # This parameter controls the steepness of the sigmoid curve.
                
                # The formula is constructed so that predictions lower than the center get a high probability,
                # and predictions higher than the center get a low probability.
                probability = torch.sigmoid(-(prediction - center) / width).item()
                
                return probability
                
        except Exception as e:
            return None
    
    def calculate_drug_properties(self, smiles: str) -> dict:
        """
        Calculates a variety of important medicinal chemistry properties for a given molecule.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return {}
            
            properties = {
                'Molecular_Weight': Descriptors.MolWt(mol),
                'LogP': Crippen.MolLogP(mol),
                'H_Bond_Donors': Lipinski.NumHDonors(mol),
                'H_Bond_Acceptors': Lipinski.NumHAcceptors(mol),
                'TPSA': Descriptors.TPSA(mol),
                'Rotatable_Bonds': Descriptors.NumRotatableBonds(mol),
                'Aromatic_Rings': Descriptors.NumAromaticRings(mol),
                # Count how many of Lipinski's rules are violated.
                'Lipinski_Violations': sum(1 for rule in Lipinski.rdMolDescriptors.GetLipinskiRuleViolations(mol) if rule[1]),
                # QED is a quantitative estimate of drug-likeness, from 0 (undesirable) to 1 (desirable).
                'QED_Score': Descriptors.qed(mol),
            }
            return properties
        except Exception as e:
            return {}
    
    def screen_compounds(self, model, df, batch_size=100, min_probability=0.7):
        """
        Screens the entire dataset of compounds for potential EGFR activity.
        """
        print(f"\n🔬 Screening {len(df):,} compounds for EGFR activity...")
        print(f"🎯 Minimum probability threshold: {min_probability}")
        
        discoveries = []
        failed_count = 0
        
        # Using tqdm provides a convenient progress bar for this long-running process.
        for i in tqdm(range(0, len(df), batch_size), desc="Screening batches"):
            batch_df = df.iloc[i:i+batch_size]
            
            for idx, row in batch_df.iterrows():
                smiles = row['SMILES']
                
                # 1. Convert SMILES to graph.
                graph_data = self.smiles_to_graph(smiles)
                if graph_data is None:
                    failed_count += 1
                    continue
                
                # 2. Predict activity.
                probability = self.predict_egfr_activity(model, graph_data)
                if probability is None:
                    failed_count += 1
                    continue
                
                # 3. If the compound is predicted to be active (a "hit"), save it.
                if probability >= min_probability:
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
    
    def get_confidence_level(self, probability: float) -> str:
        """Assigns a qualitative confidence level based on the probability score."""
        if probability >= 0.9: return "Very High"
        elif probability >= 0.8: return "High"
        elif probability >= 0.7: return "Moderate"
        else: return "Low"
    
    def rank_discoveries(self, discoveries: list) -> pd.DataFrame:
        """
        Ranks the discovered hits using a composite score that balances predicted activity
        with overall drug-likeness.
        """
        print(f"\n📊 Ranking {len(discoveries)} discoveries...")
        
        df = pd.DataFrame(discoveries)
        
        # --- Composite Drug-Likeness Score Calculation ---
        # A simple weighted average of different important properties.
        df['Drug_Likeness_Score'] = 0
        
        # 40% weight: The predicted probability of EGFR activity. This is the most important factor.
        df['Drug_Likeness_Score'] += df['EGFR_Probability'] * 0.4
        
        # 20% weight: The QED score.
        if 'QED_Score' in df.columns:
            df['Drug_Likeness_Score'] += df['QED_Score'].fillna(0) * 0.2
        
        # 20% weight: Lipinski's Rule of Five compliance. Fewer violations are better.
        if 'Lipinski_Violations' in df.columns:
            lipinski_score = (2 - df['Lipinski_Violations'].clip(0, 2)) / 2 # Normalize to 0-1 range.
            df['Drug_Likeness_Score'] += lipinski_score * 0.2
        
        # 10% weight: Preference for a certain molecular weight range (300-450 Da).
        if 'Molecular_Weight' in df.columns:
            mw_score = (1 - abs(df['Molecular_Weight'] - 375) / 375).clip(0, 1)
            df['Drug_Likeness_Score'] += mw_score.fillna(0) * 0.1
        
        # 10% weight: Preference for a certain LogP range (1-3).
        if 'LogP' in df.columns:
            logp_score = (1 - abs(df['LogP'] - 2) / 5).clip(0, 1)
            df['Drug_Likeness_Score'] += logp_score.fillna(0) * 0.1
        
        # Sort the final DataFrame by this composite score in descending order.
        df = df.sort_values('Drug_Likeness_Score', ascending=False)
        
        return df
    
    def save_discoveries(self, discoveries_df: pd.DataFrame, filename="egfr_drug_discoveries.csv"):
        """Saves the ranked discoveries to a CSV file."""
        discoveries_df.to_csv(filename, index=False)
        print(f"💾 Discoveries saved to: {filename}")
        return filename
    
    def save_top_discoveries_text(self, discoveries_df: pd.DataFrame, top_n=100, filename="top_100_discoveries.txt"):
        """Saves a detailed, human-readable summary of the top N discoveries to a text file."""
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
                for prop, fmt, unit in [
                    ('Molecular_Weight', ':.2f', ' Da'), ('LogP', ':.2f', ''), ('H_Bond_Donors', 'd', ''),
                    ('H_Bond_Acceptors', 'd', ''), ('TPSA', ':.2f', ''), ('Rotatable_Bonds', 'd', ''),
                    ('Aromatic_Rings', 'd', ''), ('QED_Score', ':.3f', ''), ('Lipinski_Violations', 'd', '')
                ]:
                    if prop in row and pd.notna(row[prop]):
                        f.write(f"    - {prop.replace('_', ' ')}: {row[prop]:{fmt}}{unit}\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
        
        print(f"✅ Top discoveries text file saved.")
    
    def print_top_discoveries(self, discoveries_df, top_n=20):
        """Prints a summary of the top N discoveries to the console."""
        print(f"\n🏆 TOP {min(top_n, len(discoveries_df))} EGFR DRUG DISCOVERIES")
        print("=" * 80)
        
        for i, (_, row) in enumerate(discoveries_df.head(top_n).iterrows()):
            print(f"\n🥇 RANK #{i+1}")
            print(f"   ChEMBL ID: {row['ChEMBL_ID']}")
            print(f"   SMILES: {row['SMILES']}")
            print(f"   EGFR Probability: {row['EGFR_Probability']:.3f} ({row['Confidence_Level']} confidence)")
            print(f"   Drug-Likeness Score: {row['Drug_Likeness_Score']:.3f}")
            print(f"   Properties:")
            if 'Molecular_Weight' in row and pd.notna(row['Molecular_Weight']):
                print(f"     • MW: {row['Molecular_Weight']:.1f} Da, LogP: {row.get('LogP', 'N/A'):.2f}, QED: {row.get('QED_Score', 'N/A'):.3f}")
    
    def run_discovery_pipeline(self, min_probability=0.7, max_compounds=100000):
        """
        Runs the complete drug discovery pipeline from start to finish.
        """
        print(f"\n🚀 STARTING EGFR CANCER DRUG DISCOVERY")
        print("=" * 60)
        
        model = self.load_model()
        if model is None: return None
        
        df = self.load_chembl_dataset()
        if df is None: return None
        
        if len(df) > max_compounds:
            print(f"📊 Limiting dataset to {max_compounds:,} compounds for faster screening")
            df = df.sample(n=max_compounds, random_state=42)
        
        discoveries = self.screen_compounds(model, df, min_probability=min_probability)
        if not discoveries:
            print("\n😞 No promising compounds found. Consider lowering the probability threshold.")
            return None
        
        discoveries_df = self.rank_discoveries(discoveries)
        
        output_path = self.save_discoveries(discoveries_df)
        self.save_top_discoveries_text(discoveries_df, top_n=100)
        self.print_top_discoveries(discoveries_df)
        
        print(f"\n🎉 DRUG DISCOVERY COMPLETE!")
        print(f"💊 Total discoveries: {len(discoveries_df)}")
        print(f"📁 Results saved to: {output_path}")
        print(f"🔬 Ready for experimental validation!")
        
        return discoveries_df

def main():
    """Main execution function."""
    discovery = EGFRDrugDiscovery(
        model_path="egfr_model.pth",
        dataset_path="drug_discovery_dataset/chembl_drug_discovery_dataset.csv"
    )
    
    results = discovery.run_discovery_pipeline(
        min_probability=0.7,
        max_compounds=100000
    )
    
    if results is not None:
        print(f"\n🎯 SUCCESS! Discovered {len(results)} potential EGFR inhibitors!")
        print("🔬 Next steps:")
        print("   1. Review top candidates in 'egfr_drug_discoveries.csv' and 'top_100_discoveries.txt'")
        print("   2. Perform molecular docking studies for the best hits.")
        print("   3. Consider experimental validation for the most promising compounds.")
        print("   4. Check literature for known activities or similar scaffolds.")

if __name__ == "__main__":
    main()
