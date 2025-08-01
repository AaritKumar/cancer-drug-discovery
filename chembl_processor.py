#!/usr/bin/env python3
"""
ChEMBL Drug Discovery Dataset Processor
Processes ChEMBL chemreps.txt file to create drug discovery dataset

Usage: python chembl_processor.py
Make sure 'chembl_35_chemreps.txt' is in the same directory
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import sys

class ChEMBLProcessor:
    def __init__(self, chembl_file="chembl_35_chemreps.txt"):
        self.chembl_file = chembl_file
        self.output_dir = "drug_discovery_dataset"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Drug-like filtering criteria (Lipinski's Rule of Five + extensions)
        self.drug_filters = {
            'MW': [150, 500],           # Molecular weight 150-500 Da
            'LogP': [-0.4, 5.6],        # Lipophilicity (ALogP or XLogP)
            'HBD': [0, 5],              # Hydrogen bond donors ≤5
            'HBA': [0, 10],             # Hydrogen bond acceptors ≤10
            'TPSA': [0, 140],           # Topological polar surface area ≤140
            'RotBond': [0, 10]          # Rotatable bonds ≤10
        }
    
    def load_chembl_data(self) -> pd.DataFrame:
        """Load and parse ChEMBL chemreps file"""
        print(f"📥 Loading ChEMBL data from {self.chembl_file}...")
        
        if not os.path.exists(self.chembl_file):
            print(f"❌ File not found: {self.chembl_file}")
            print("💡 Make sure the ChEMBL file is in the same directory as this script")
            sys.exit(1)
        
        try:
            # ChEMBL chemreps files are typically tab-separated
            df = pd.read_csv(self.chembl_file, sep='\t', low_memory=False)
            print(f"✅ Loaded {len(df):,} compounds from ChEMBL")
            
            # Display column information
            print(f"\n📊 Available columns:")
            for i, col in enumerate(df.columns):
                print(f"   {i+1:2d}. {col}")
            
            print(f"\n📏 Data shape: {df.shape}")
            return df
            
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            print("💡 Trying different separators...")
            
            # Try comma-separated
            try:
                df = pd.read_csv(self.chembl_file, sep=',', low_memory=False)
                print(f"✅ Loaded {len(df):,} compounds (comma-separated)")
                return df
            except:
                print("❌ Could not parse file with common separators")
                sys.exit(1)
    
    def identify_columns(self, df: pd.DataFrame) -> dict:
        """Identify relevant columns for drug-like filtering"""
        print("\n🔍 Identifying relevant columns...")
        
        column_mapping = {}
        
        # Common column name patterns
        patterns = {
            'smiles': ['canonical_smiles', 'smiles', 'structure', 'mol_structure'],
            'molecular_weight': ['molecular_weight', 'mw', 'molweight', 'mol_wt'],
            'logp': ['alogp', 'xlogp', 'logp', 'clogp'],
            'hbd': ['hbd', 'num_hbd', 'hbond_donor', 'hydrogen_bond_donor'],
            'hba': ['hba', 'num_hba', 'hbond_acceptor', 'hydrogen_bond_acceptor'],
            'tpsa': ['tpsa', 'psa', 'polar_surface_area'],
            'rotatable_bonds': ['rtb', 'num_rtb', 'rotatable_bonds', 'rotbond'],
            'compound_id': ['chembl_id', 'compound_id', 'molregno', 'id']
        }
        
        # Match columns to patterns
        for prop, pattern_list in patterns.items():
            for col in df.columns:
                col_lower = col.lower().replace('_', '').replace(' ', '')
                for pattern in pattern_list:
                    pattern_clean = pattern.lower().replace('_', '')
                    if pattern_clean in col_lower:
                        column_mapping[prop] = col
                        print(f"   ✅ {prop}: {col}")
                        break
                if prop in column_mapping:
                    break
        
        return column_mapping
    
    def apply_drug_like_filters(self, df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
        """Apply drug-like property filters"""
        print("\n🔬 Applying drug-like filters...")
        
        original_count = len(df)
        filtered_df = df.copy()
        
        # Apply each filter
        filter_stats = {}
        
        # Molecular Weight filter
        if 'molecular_weight' in col_map:
            mw_col = col_map['molecular_weight']
            filtered_df[mw_col] = pd.to_numeric(filtered_df[mw_col], errors='coerce')
            before_count = len(filtered_df)
            filtered_df = filtered_df[
                (filtered_df[mw_col] >= self.drug_filters['MW'][0]) & 
                (filtered_df[mw_col] <= self.drug_filters['MW'][1])
            ]
            filter_stats['Molecular Weight (150-500 Da)'] = f"{before_count:,} → {len(filtered_df):,}"
        
        # LogP filter
        if 'logp' in col_map:
            logp_col = col_map['logp']
            filtered_df[logp_col] = pd.to_numeric(filtered_df[logp_col], errors='coerce')
            before_count = len(filtered_df)
            filtered_df = filtered_df[
                (filtered_df[logp_col] >= self.drug_filters['LogP'][0]) & 
                (filtered_df[logp_col] <= self.drug_filters['LogP'][1])
            ]
            filter_stats['LogP (-0.4 to 5.6)'] = f"{before_count:,} → {len(filtered_df):,}"
        
        # Hydrogen Bond Donors filter
        if 'hbd' in col_map:
            hbd_col = col_map['hbd']
            filtered_df[hbd_col] = pd.to_numeric(filtered_df[hbd_col], errors='coerce')
            before_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df[hbd_col] <= self.drug_filters['HBD'][1]]
            filter_stats['H-bond Donors (≤5)'] = f"{before_count:,} → {len(filtered_df):,}"
        
        # Hydrogen Bond Acceptors filter
        if 'hba' in col_map:
            hba_col = col_map['hba']
            filtered_df[hba_col] = pd.to_numeric(filtered_df[hba_col], errors='coerce')
            before_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df[hba_col] <= self.drug_filters['HBA'][1]]
            filter_stats['H-bond Acceptors (≤10)'] = f"{before_count:,} → {len(filtered_df):,}"
        
        # TPSA filter
        if 'tpsa' in col_map:
            tpsa_col = col_map['tpsa']
            filtered_df[tpsa_col] = pd.to_numeric(filtered_df[tpsa_col], errors='coerce')
            before_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df[tpsa_col] <= self.drug_filters['TPSA'][1]]
            filter_stats['TPSA (≤140)'] = f"{before_count:,} → {len(filtered_df):,}"
        
        # Rotatable Bonds filter
        if 'rotatable_bonds' in col_map:
            rtb_col = col_map['rotatable_bonds']
            filtered_df[rtb_col] = pd.to_numeric(filtered_df[rtb_col], errors='coerce')
            before_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df[rtb_col] <= self.drug_filters['RotBond'][1]]
            filter_stats['Rotatable Bonds (≤10)'] = f"{before_count:,} → {len(filtered_df):,}"
        
        # Remove rows with missing SMILES
        if 'smiles' in col_map:
            smiles_col = col_map['smiles']
            before_count = len(filtered_df)
            filtered_df = filtered_df.dropna(subset=[smiles_col])
            filtered_df = filtered_df[filtered_df[smiles_col].str.len() > 5]  # Valid SMILES
            filter_stats['Valid SMILES'] = f"{before_count:,} → {len(filtered_df):,}"
        
        # Print filtering results
        print(f"\n📊 Filtering Results:")
        for filter_name, stats in filter_stats.items():
            print(f"   {filter_name}: {stats}")
        
        print(f"\n🎯 Final Result: {original_count:,} → {len(filtered_df):,} compounds ({len(filtered_df)/original_count*100:.1f}% retained)")
        
        return filtered_df
    
    def prepare_discovery_dataset(self, df: pd.DataFrame, col_map: dict, max_compounds: int = 100000) -> pd.DataFrame:
        """Prepare final dataset for drug discovery"""
        print(f"\n🎯 Preparing drug discovery dataset (max {max_compounds:,} compounds)...")
        
        # Select essential columns
        essential_cols = []
        col_rename = {}
        
        if 'compound_id' in col_map:
            essential_cols.append(col_map['compound_id'])
            col_rename[col_map['compound_id']] = 'ChEMBL_ID'
        
        if 'smiles' in col_map:
            essential_cols.append(col_map['smiles'])
            col_rename[col_map['smiles']] = 'SMILES'
        
        if 'molecular_weight' in col_map:
            essential_cols.append(col_map['molecular_weight'])
            col_rename[col_map['molecular_weight']] = 'Molecular_Weight'
        
        if 'logp' in col_map:
            essential_cols.append(col_map['logp'])
            col_rename[col_map['logp']] = 'LogP'
        
        if 'hbd' in col_map:
            essential_cols.append(col_map['hbd'])
            col_rename[col_map['hbd']] = 'H_Bond_Donors'
        
        if 'hba' in col_map:
            essential_cols.append(col_map['hba'])
            col_rename[col_map['hba']] = 'H_Bond_Acceptors'
        
        if 'tpsa' in col_map:
            essential_cols.append(col_map['tpsa'])
            col_rename[col_map['tpsa']] = 'TPSA'
        
        if 'rotatable_bonds' in col_map:
            essential_cols.append(col_map['rotatable_bonds'])
            col_rename[col_map['rotatable_bonds']] = 'Rotatable_Bonds'
        
        # Create final dataset
        final_df = df[essential_cols].copy()
        final_df = final_df.rename(columns=col_rename)
        
        # Sample if we have too many compounds
        if len(final_df) > max_compounds:
            print(f"📊 Randomly sampling {max_compounds:,} compounds from {len(final_df):,} available")
            final_df = final_df.sample(n=max_compounds, random_state=42)
        
        # Remove duplicates based on SMILES
        if 'SMILES' in final_df.columns:
            before_dedup = len(final_df)
            final_df = final_df.drop_duplicates(subset=['SMILES'])
            print(f"🔄 Removed {before_dedup - len(final_df):,} duplicate SMILES")
        
        return final_df
    
    def save_datasets(self, df: pd.DataFrame):
        """Save processed datasets in multiple formats"""
        print(f"\n💾 Saving datasets...")
        
        # Save full dataset
        full_path = os.path.join(self.output_dir, "chembl_drug_discovery_dataset.csv")
        df.to_csv(full_path, index=False)
        print(f"✅ Full dataset: {full_path}")
        
        # Save SMILES-only file for quick model input
        if 'SMILES' in df.columns:
            smiles_path = os.path.join(self.output_dir, "drug_discovery_smiles.txt")
            with open(smiles_path, 'w') as f:
                for smiles in df['SMILES']:
                    f.write(f"{smiles}\n")
            print(f"✅ SMILES file: {smiles_path}")
        
        # Save compact dataset for model testing
        if 'SMILES' in df.columns and 'ChEMBL_ID' in df.columns:
            compact_df = df[['ChEMBL_ID', 'SMILES']].copy()
            compact_path = os.path.join(self.output_dir, "drug_discovery_compact.csv")
            compact_df.to_csv(compact_path, index=False)
            print(f"✅ Compact dataset: {compact_path}")
        
        return full_path
    
    def print_dataset_summary(self, df: pd.DataFrame):
        """Print comprehensive dataset summary"""
        print(f"\n📊 FINAL DATASET SUMMARY")
        print("=" * 50)
        print(f"Total compounds: {len(df):,}")
        
        if 'SMILES' in df.columns:
            print(f"Unique SMILES: {df['SMILES'].nunique():,}")
        
        print(f"\n🧪 MOLECULAR PROPERTY STATISTICS:")
        numeric_cols = ['Molecular_Weight', 'LogP', 'H_Bond_Donors', 'H_Bond_Acceptors', 'TPSA', 'Rotatable_Bonds']
        
        for col in numeric_cols:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce')
                print(f"   {col}:")
                print(f"     Range: {series.min():.2f} - {series.max():.2f}")
                print(f"     Mean:  {series.mean():.2f} ± {series.std():.2f}")
        
        print(f"\n🎯 READY FOR EGFR MODEL TESTING!")
        print(f"🔬 Next step: Load SMILES and run your cancer drug predictions!")

def main():
    """Main processing function"""
    print("🚀 ChEMBL Drug Discovery Dataset Processor")
    print("=" * 50)
    
    # Initialize processor
    processor = ChEMBLProcessor()
    
    # Load ChEMBL data
    df = processor.load_chembl_data()
    
    # Identify columns
    col_map = processor.identify_columns(df)
    
    if not col_map:
        print("❌ Could not identify essential columns. Please check file format.")
        return
    
    # Apply drug-like filters
    filtered_df = processor.apply_drug_like_filters(df, col_map)
    
    if len(filtered_df) == 0:
        print("❌ No compounds passed filters. Try relaxing criteria.")
        return
    
    # Prepare final dataset
    final_df = processor.prepare_discovery_dataset(filtered_df, col_map)
    
    # Save datasets
    output_path = processor.save_datasets(final_df)
    
    # Print summary
    processor.print_dataset_summary(final_df)
    
    print(f"\n🎉 SUCCESS!")
    print(f"📁 Dataset saved to: {output_path}")
    print(f"🧬 Ready for EGFR model testing with {len(final_df):,} drug-like compounds!")

if __name__ == "__main__":
    main()