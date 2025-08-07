#!/usr/bin/env python3
"""
ChEMBL Drug Discovery Dataset Processor
=======================================

This script is designed to process a large-scale chemical data file from the ChEMBL database,
specifically the 'chemreps.txt' file, which contains comprehensive information about millions 
of chemical compounds. 

The primary goal is to filter this vast dataset to create a smaller, more focused collection
of "drug-like" molecules that can be used for virtual screening in a drug discovery pipeline.
This is a critical first step in computational drug discovery, as it narrows down the search
space from millions of compounds to a more manageable number of high-quality candidates.

The script performs the following key steps:
1.  Loads the raw ChEMBL data file.
2.  Identifies the relevant columns for molecular properties (e.g., molecular weight, LogP).
3.  Applies a series of filters based on established medicinal chemistry principles, such as
    Lipinski's Rule of Five, to select for compounds with favorable drug-like properties.
4.  Prepares and saves the final, filtered dataset in multiple formats for use in downstream
    modeling and analysis.

Usage:
    python chembl_processor.py
    
Prerequisites:
    - Make sure the 'chembl_35_chemreps.txt' file is located in the same directory as this script.
    - Python libraries required: pandas, numpy, tqdm.
"""

# Import necessary libraries
import pandas as pd  # For data manipulation and reading CSV files
import numpy as np   # For numerical operations (not heavily used here, but good practice)
import os            # For interacting with the operating system, like creating directories
from tqdm import tqdm # Provides a progress bar for long-running loops, enhancing user experience
import sys           # For system-specific parameters and functions, like exiting the script

class ChEMBLProcessor:
    """
    A class to handle the processing of the ChEMBL chemical representations file.
    It encapsulates all the logic for loading, filtering, and saving the dataset.
    """
    
    def __init__(self, chembl_file="chembl_35_chemreps.txt"):
        """
        Initializes the ChEMBLProcessor.

        Args:
            chembl_file (str): The name of the input ChEMBL data file.
        """
        # Store the path to the input ChEMBL file
        self.chembl_file = chembl_file
        # Define the directory where the processed datasets will be saved
        self.output_dir = "drug_discovery_dataset"
        
        # Automatically create the output directory if it doesn't already exist.
        # The `exist_ok=True` argument prevents an error from being raised if the directory is already there.
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define the criteria for what constitutes a "drug-like" molecule.
        # These filters are based on common medicinal chemistry rules, most notably Lipinski's Rule of Five,
        # which predict if a compound has properties that would make it a likely orally active drug in humans.
        self.drug_filters = {
            'MW': [150, 500],           # Molecular Weight: Not too small, not too large. Range is 150-500 Daltons.
            'LogP': [-0.4, 5.6],        # Lipophilicity (LogP): A measure of how well a compound dissolves in fats/oils.
                                        # This range balances solubility and permeability.
            'HBD': [0, 5],              # Hydrogen Bond Donors: Number of hydrogen atoms bonded to electronegative atoms (O, N).
                                        # Should be 5 or fewer.
            'HBA': [0, 10],             # Hydrogen Bond Acceptors: Number of electronegative atoms (O, N).
                                        # Should be 10 or fewer.
            'TPSA': [0, 140],           # Topological Polar Surface Area: An indicator of a molecule's ability to permeate cell membranes.
                                        # Should be 140 Å² or less.
            'RotBond': [0, 10]          # Rotatable Bonds: Number of bonds that can rotate freely. Too many can lead to poor bioavailability.
                                        # Should be 10 or fewer.
        }
    
    def load_chembl_data(self) -> pd.DataFrame:
        """
        Loads and parses the ChEMBL chemreps file into a pandas DataFrame.
        It attempts to handle different file separators (tab or comma).

        Returns:
            pd.DataFrame: A DataFrame containing the loaded ChEMBL data.
        """
        print(f"📥 Loading ChEMBL data from {self.chembl_file}...")
        
        # Check if the required ChEMBL file actually exists before trying to load it.
        if not os.path.exists(self.chembl_file):
            print(f"❌ File not found: {self.chembl_file}")
            print("💡 Make sure the ChEMBL file is in the same directory as this script")
            # Exit the script if the file is not found, as processing cannot continue.
            sys.exit(1)
        
        try:
            # The standard ChEMBL chemreps file is tab-separated. We attempt to load it this way first.
            # `low_memory=False` is used to load the entire file into memory at once, which can be faster for large files,
            # though it requires more RAM.
            df = pd.read_csv(self.chembl_file, sep='\t', low_memory=False)
            print(f"✅ Loaded {len(df):,} compounds from ChEMBL")
            
            # Display information about the loaded data to the user.
            print(f"\n📊 Available columns:")
            for i, col in enumerate(df.columns):
                print(f"   {i+1:2d}. {col}")
            
            print(f"\n📏 Data shape: {df.shape}")
            return df
            
        except Exception as e:
            # If loading with a tab separator fails, we catch the error and try a comma separator,
            # which is another common format for data files.
            print(f"❌ Error loading file: {e}")
            print("💡 Trying different separators...")
            
            try:
                df = pd.read_csv(self.chembl_file, sep=',', low_memory=False)
                print(f"✅ Loaded {len(df):,} compounds (comma-separated)")
                return df
            except:
                # If both attempts fail, we inform the user and exit the script.
                print("❌ Could not parse file with common separators")
                sys.exit(1)
    
    def identify_columns(self, df: pd.DataFrame) -> dict:
        """
        Automatically identifies the relevant columns for drug-like filtering by matching
        common naming patterns. This makes the script more robust to variations in column names
        across different versions of the ChEMBL file.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            dict: A mapping from a standard property name (e.g., 'molecular_weight') to the
                  actual column name found in the DataFrame (e.g., 'mol_wt').
        """
        print("\n🔍 Identifying relevant columns...")
        
        column_mapping = {}
        
        # A dictionary of common naming patterns for each molecular property.
        # This allows the script to find the correct column even if it's named 'mw' instead of 'molecular_weight'.
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
        
        # Iterate through the patterns and DataFrame columns to find matches.
        for prop, pattern_list in patterns.items():
            for col in df.columns:
                # Standardize column names for case-insensitive and flexible matching.
                col_lower = col.lower().replace('_', '').replace(' ', '')
                for pattern in pattern_list:
                    pattern_clean = pattern.lower().replace('_', '')
                    if pattern_clean in col_lower:
                        # If a match is found, store it in the mapping and break to the next property.
                        column_mapping[prop] = col
                        print(f"   ✅ {prop}: {col}")
                        break
                if prop in column_mapping:
                    break
        
        return column_mapping
    
    def apply_drug_like_filters(self, df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
        """
        Applies the predefined drug-like property filters to the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be filtered.
            col_map (dict): The column mapping created by `identify_columns`.

        Returns:
            pd.DataFrame: The filtered DataFrame containing only drug-like compounds.
        """
        print("\n🔬 Applying drug-like filters...")
        
        original_count = len(df)
        filtered_df = df.copy()
        
        filter_stats = {}
        
        # Sequentially apply each filter and record how many compounds pass each stage.
        # This provides a clear report of how the dataset is being reduced.

        # Molecular Weight filter
        if 'molecular_weight' in col_map:
            mw_col = col_map['molecular_weight']
            # Ensure the column is numeric, converting any non-numeric values to NaN (Not a Number).
            filtered_df[mw_col] = pd.to_numeric(filtered_df[mw_col], errors='coerce')
            before_count = len(filtered_df)
            # Apply the filter based on the min and max values defined in `self.drug_filters`.
            filtered_df = filtered_df[
                (filtered_df[mw_col] >= self.drug_filters['MW'][0]) & 
                (filtered_df[mw_col] <= self.drug_filters['MW'][1])
            ]
            filter_stats['Molecular Weight (150-500 Da)'] = f"{before_count:,} → {len(filtered_df):,}"
        
        # LogP filter (similar logic as above)
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
        
        # Final cleanup: Remove any rows that are missing a SMILES string, which is essential for modeling.
        if 'smiles' in col_map:
            smiles_col = col_map['smiles']
            before_count = len(filtered_df)
            filtered_df = filtered_df.dropna(subset=[smiles_col])
            # Also remove SMILES that are too short to be meaningful molecules.
            filtered_df = filtered_df[filtered_df[smiles_col].str.len() > 5]
            filter_stats['Valid SMILES'] = f"{before_count:,} → {len(filtered_df):,}"
        
        # Print a summary of the filtering results to the console.
        print(f"\n📊 Filtering Results:")
        for filter_name, stats in filter_stats.items():
            print(f"   {filter_name}: {stats}")
        
        print(f"\n🎯 Final Result: {original_count:,} → {len(filtered_df):,} compounds ({len(filtered_df)/original_count*100:.1f}% retained)")
        
        return filtered_df
    
    def prepare_discovery_dataset(self, df: pd.DataFrame, col_map: dict, max_compounds: int = 100000) -> pd.DataFrame:
        """
        Prepares the final, cleaned dataset for drug discovery by selecting and renaming
        essential columns and sampling if necessary.

        Args:
            df (pd.DataFrame): The filtered DataFrame.
            col_map (dict): The column mapping.
            max_compounds (int): The maximum number of compounds to include in the final dataset.

        Returns:
            pd.DataFrame: The final, analysis-ready dataset.
        """
        print(f"\n🎯 Preparing drug discovery dataset (max {max_compounds:,} compounds)...")
        
        # Select only the columns that are essential for the downstream drug discovery model.
        essential_cols = []
        # Create a dictionary to rename the columns to a standard format.
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
        
        # Create the final DataFrame with selected and renamed columns.
        final_df = df[essential_cols].copy()
        final_df = final_df.rename(columns=col_rename)
        
        # If the dataset is still larger than our desired maximum, take a random sample.
        # This is useful for creating a manageable dataset for faster screening.
        if len(final_df) > max_compounds:
            print(f"📊 Randomly sampling {max_compounds:,} compounds from {len(final_df):,} available")
            final_df = final_df.sample(n=max_compounds, random_state=42)
        
        # Final check for duplicates based on the SMILES string, as a single molecule can have multiple ChEMBL IDs.
        if 'SMILES' in final_df.columns:
            before_dedup = len(final_df)
            final_df = final_df.drop_duplicates(subset=['SMILES'])
            print(f"🔄 Removed {before_dedup - len(final_df):,} duplicate SMILES")
        
        return final_df
    
    def save_datasets(self, df: pd.DataFrame):
        """
        Saves the processed dataset in multiple formats to the output directory.

        Args:
            df (pd.DataFrame): The final DataFrame to save.
        """
        print(f"\n💾 Saving datasets...")
        
        # 1. Save the full dataset with all columns as a CSV file.
        full_path = os.path.join(self.output_dir, "chembl_drug_discovery_dataset.csv")
        df.to_csv(full_path, index=False)
        print(f"✅ Full dataset: {full_path}")
        
        # 2. Save a text file containing only the SMILES strings. This is often a convenient format
        #    for direct input into modeling scripts that only need the molecular structure.
        if 'SMILES' in df.columns:
            smiles_path = os.path.join(self.output_dir, "drug_discovery_smiles.txt")
            with open(smiles_path, 'w') as f:
                for smiles in df['SMILES']:
                    f.write(f"{smiles}\n")
            print(f"✅ SMILES file: {smiles_path}")
        
        # 3. Save a compact version of the dataset with just the compound ID and SMILES.
        #    This is useful for quick lookups or sharing a smaller file.
        if 'SMILES' in df.columns and 'ChEMBL_ID' in df.columns:
            compact_df = df[['ChEMBL_ID', 'SMILES']].copy()
            compact_path = os.path.join(self.output_dir, "drug_discovery_compact.csv")
            compact_df.to_csv(compact_path, index=False)
            print(f"✅ Compact dataset: {compact_path}")
        
        return full_path
    
    def print_dataset_summary(self, df: pd.DataFrame):
        """
        Prints a comprehensive summary of the final dataset, including statistics
        for the key molecular properties.

        Args:
            df (pd.DataFrame): The final DataFrame.
        """
        print(f"\n📊 FINAL DATASET SUMMARY")
        print("=" * 50)
        print(f"Total compounds: {len(df):,}")
        
        if 'SMILES' in df.columns:
            print(f"Unique SMILES: {df['SMILES'].nunique():,}")
        
        print(f"\n🧪 MOLECULAR PROPERTY STATISTICS:")
        numeric_cols = ['Molecular_Weight', 'LogP', 'H_Bond_Donors', 'H_Bond_Acceptors', 'TPSA', 'Rotatable_Bonds']
        
        # Calculate and print descriptive statistics for each property.
        for col in numeric_cols:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce')
                print(f"   {col}:")
                print(f"     Range: {series.min():.2f} - {series.max():.2f}")
                print(f"     Mean:  {series.mean():.2f} ± {series.std():.2f}")
        
        print(f"\n🎯 READY FOR EGFR MODEL TESTING!")
        print(f"🔬 Next step: Load SMILES and run your cancer drug predictions!")

def main():
    """
    The main function that orchestrates the entire data processing pipeline.
    """
    print("🚀 ChEMBL Drug Discovery Dataset Processor")
    print("=" * 50)
    
    # Initialize the processor class.
    processor = ChEMBLProcessor()
    
    # Step 1: Load the raw data.
    df = processor.load_chembl_data()
    
    # Step 2: Identify the relevant columns.
    col_map = processor.identify_columns(df)
    
    if not col_map:
        print("❌ Could not identify essential columns. Please check file format.")
        return
    
    # Step 3: Apply the drug-like filters.
    filtered_df = processor.apply_drug_like_filters(df, col_map)
    
    if len(filtered_df) == 0:
        print("❌ No compounds passed filters. Try relaxing criteria.")
        return
    
    # Step 4: Prepare the final dataset for discovery.
    final_df = processor.prepare_discovery_dataset(filtered_df, col_map)
    
    # Step 5: Save the processed datasets.
    output_path = processor.save_datasets(final_df)
    
    # Step 6: Print a final summary.
    processor.print_dataset_summary(final_df)
    
    print(f"\n🎉 SUCCESS!")
    print(f"📁 Dataset saved to: {output_path}")
    print(f"🧬 Ready for EGFR model testing with {len(final_df):,} drug-like compounds!")

# This standard Python construct ensures that the main() function is called only when the script is executed directly.
if __name__ == "__main__":
    main()
