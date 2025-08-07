"""
Large-Scale EGFR Bioactivity Dataset Download Script
====================================================

This script is responsible for downloading a comprehensive dataset of bioactivity data
for the Epidermal Growth Factor Receptor (EGFR) from the ChEMBL database. EGFR is a key
target in cancer therapy, and having a large, high-quality dataset of molecules that
interact with it is essential for training robust machine learning models.

The script performs several key functions:
1.  **Connects to the ChEMBL Database**: It uses the official `chembl_webresource_client`
    to programmatically access the ChEMBL database.
2.  **Downloads Bioactivity Data**: It queries the database for all compounds tested against
    EGFR (ChEMBL ID: CHEMBL203) and retrieves their bioactivity measurements (e.g., IC50, Ki).
3.  **Downloads Molecular Structures**: For the identified compounds, it downloads their
    chemical structures (as SMILES strings) and key molecular properties.
4.  **Cleans and Processes Data**: The raw data is cleaned and standardized. This includes:
    - Converting different activity units (nM, uM) to a consistent scale (nM).
    - Handling ambiguous measurements (e.g., values reported as '<' or '>').
    - Removing duplicate entries and prioritizing the most reliable data points.
5.  **Applies Final Quality Filters**: It applies drug-likeness filters (e.g., Lipinski's Rule of Five)
    to ensure the resulting dataset consists of high-quality, drug-like molecules.
6.  **Saves the Final Dataset**: The cleaned and processed data is saved to a CSV file,
    ready to be used for training a Graph Neural Network (GNN) model.
7.  **Provides Validation**: It runs a series of checks on the final dataset to ensure its
    quality and readiness for machine learning.

This automated pipeline ensures the creation of a large, consistent, and high-quality dataset
for training advanced predictive models for cancer drug discovery.
"""

# Import necessary libraries
import pandas as pd  # For data manipulation and creating DataFrames
import numpy as np   # For numerical operations, especially for handling logarithms and NaN values
from chembl_webresource_client.new_client import new_client  # The official client to interact with the ChEMBL API
import time          # To pause the script periodically to avoid overwhelming the ChEMBL server
import os            # To check for file existence and manage file paths
from datetime import datetime  # To timestamp the start of the script execution

def download_large_egfr_dataset():
    """
    Downloads a comprehensive set of EGFR bioactivity data directly from the ChEMBL database.
    This function uses broad filters to capture as much relevant data as possible.
    
    Returns:
        pd.DataFrame: A DataFrame containing the raw bioactivity data.
    """
    
    print("🚀 LARGE-SCALE EGFR DATASET COLLECTION")
    print("=" * 45)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize the ChEMBL API clients for different data types.
    activity = new_client.activity
    molecule = new_client.molecule
    target = new_client.target
    
    # First, we verify that our target, EGFR, exists and retrieve its official name.
    # The ChEMBL ID for human EGFR is 'CHEMBL203'.
    print("🎯 Verifying EGFR target...")
    egfr_target = target.get('CHEMBL203')
    print(f"Target: {egfr_target['pref_name']}")
    
    # Now, we download the bioactivity data.
    print("📥 Downloading comprehensive EGFR bioactivity data...")
    print("This will take 30-60 minutes for large dataset...")
    
    # We use broad filters to maximize the amount of data we retrieve.
    activities = activity.filter(
        target_chembl_id='CHEMBL203',                               # The specific target we are interested in.
        standard_type__in=['IC50', 'EC50', 'Ki', 'Kd', 'GI50'],      # Common measures of bioactivity.
        standard_relation__in=['=', '<', '<=', '>', '>='],          # Include exact and approximate measurements.
        standard_units__in=['nM', 'uM', 'pM'],                      # Allow for different concentration units.
        confidence_score__gte=5,                                    # A measure of data quality; 5 or higher is generally good.
        pchembl_value__isnull=False                                 # Ensure that a standardized activity value exists.
    )
    
    # The download can be very large, so we process it record by record and provide progress updates.
    print("🔄 Processing activities (this takes time)...")
    activities_list = []
    count = 0
    
    # Iterate through the downloaded activities.
    for activity_data in activities:
        activities_list.append(activity_data)
        count += 1
        
        # Print a progress update every 500 records.
        if count % 500 == 0:
            print(f"   Downloaded {count} records...")
            
        # To prevent data loss on long runs, we save intermediate results every 2000 records.
        if count % 2000 == 0:
            temp_df = pd.DataFrame(activities_list)
            temp_df.to_csv(f'temp_activities_{count}.csv', index=False)
            print(f"   Saved intermediate file: temp_activities_{count}.csv")
        
        # We pause for 2 seconds every 1000 records to be respectful of the ChEMBL server.
        if count % 1000 == 0:
            time.sleep(2)
    
    print(f"✅ Downloaded {len(activities_list)} activity records")
    
    # Convert the final list of dictionaries into a pandas DataFrame and save it.
    df_activities = pd.DataFrame(activities_list)
    df_activities.to_csv('all_egfr_activities.csv', index=False)
    print(f"💾 Saved all activities to: all_egfr_activities.csv")
    
    return df_activities

def process_and_clean_large_dataset(df_activities):
    """
    Cleans and standardizes the raw bioactivity data. This involves converting units,
    handling inexact measurements, and removing duplicates.

    Args:
        df_activities (pd.DataFrame): The raw DataFrame from `download_large_egfr_dataset`.

    Returns:
        pd.DataFrame: A cleaned and processed DataFrame with one entry per unique compound.
    """
    
    print("\n🧹 PROCESSING AND CLEANING LARGE DATASET")
    print("=" * 45)
    
    print(f"Starting with {len(df_activities)} raw activity records")
    
    # --- Step 1: Standardize Units ---
    print("🔧 Cleaning and standardizing data...")
    
    # Helper function to convert all activity values to nanomolar (nM), a standard unit.
    def convert_to_nm(value, unit):
        if pd.isna(value) or pd.isna(unit):
            return np.nan
        
        unit = str(unit).lower()
        if 'nm' in unit:
            return float(value)
        elif 'um' in unit or 'μm' in unit: # Handle both 'um' and the micro symbol 'μm'.
            return float(value) * 1000  # Convert micromolar (μM) to nanomolar (nM).
        elif 'pm' in unit:
            return float(value) / 1000  # Convert picomolar (pM) to nanomolar (nM).
        else:
            return np.nan
    
    # Apply this conversion to create a new, standardized column.
    df_activities['standard_value_nm'] = df_activities.apply(
        lambda row: convert_to_nm(row['standard_value'], row['standard_units']), axis=1
    )
    
    # Filter out compounds with biologically unreasonable activity values.
    df_clean = df_activities[
        (df_activities['standard_value_nm'] >= 0.1) & 
        (df_activities['standard_value_nm'] <= 1000000)
    ].copy()
    
    print(f"After unit conversion and range filtering: {len(df_clean)} records")
    
    # --- Step 2: Handle Relational Data (e.g., '<', '>') ---
    # For values like '<100', we make a conservative estimate. This is a common practice
    # in cheminformatics to retain as much data as possible.
    def handle_relations(row):
        value = row['standard_value_nm']
        relation = row.get('standard_relation', '=')
        
        if pd.isna(value): return np.nan
        
        if relation == '<': return value / 2  # A conservative estimate for 'less than'.
        elif relation == '<=': return value
        elif relation == '>': return value * 2  # A conservative estimate for 'greater than'.
        elif relation == '>=': return value
        else: return value # Assumed to be '='.
    
    df_clean['activity_value_nm'] = df_clean.apply(handle_relations, axis=1)
    
    # --- Step 3: Remove Duplicates ---
    print("🔄 Removing duplicates and selecting best measurements...")
    
    # A single compound can have multiple activity measurements. We need to select the most reliable one.
    # First, we sort by molecule, then by activity type, and finally by data confidence score (highest first).
    df_dedup = df_clean.sort_values(['molecule_chembl_id', 'standard_type', 'confidence_score'], 
                                   ascending=[True, True, False])
    # Then, we group by molecule and activity type and take the first entry, which will be the one with the highest confidence.
    df_dedup = df_dedup.groupby(['molecule_chembl_id', 'standard_type']).first().reset_index()
    
    print(f"After deduplication: {len(df_dedup)} records")
    
    # --- Step 4: Prioritize Activity Type ---
    # If a molecule still has multiple entries (e.g., both an IC50 and a Ki value), we prioritize
    # the most common and direct measure of activity, typically IC50.
    activity_priority = {'IC50': 1, 'Ki': 2, 'EC50': 3, 'Kd': 4, 'GI50': 5}
    df_dedup['priority'] = df_dedup['standard_type'].map(activity_priority).fillna(10)
    
    # We sort by molecule and then by our defined priority, taking the first (highest priority) entry.
    df_final = df_dedup.sort_values(['molecule_chembl_id', 'priority']).groupby('molecule_chembl_id').first().reset_index()
    
    print(f"After activity type prioritization: {len(df_final)} unique compounds")
    
    return df_final

def download_molecular_structures_batch(df_final):
    """
    Downloads the molecular structures (SMILES) and properties for the cleaned list
    of unique compounds. This is done in batches to be efficient and avoid server timeouts.

    Args:
        df_final (pd.DataFrame): The cleaned DataFrame of unique compounds.

    Returns:
        pd.DataFrame: A DataFrame containing molecular structures and properties.
    """
    
    print("\n🧬 DOWNLOADING MOLECULAR STRUCTURES (BATCH MODE)")
    print("=" * 50)
    
    molecule = new_client.molecule
    unique_molecules = df_final['molecule_chembl_id'].unique()
    total_molecules = len(unique_molecules)
    
    print(f"Processing {total_molecules} unique molecules...")
    
    # We process the downloads in batches to avoid sending a single, massive request to the server.
    batch_size = 100
    all_molecule_data = []
    failed_molecules = []
    
    for i in range(0, total_molecules, batch_size):
        batch_end = min(i + batch_size, total_molecules)
        batch_molecules = unique_molecules[i:batch_end]
        
        print(f"Processing batch {i//batch_size + 1}/{(total_molecules-1)//batch_size + 1} "
              f"({i+1}-{batch_end}/{total_molecules})")
        
        batch_data = []
        
        for mol_id in batch_molecules:
            try:
                mol_info = molecule.get(mol_id)
                
                if mol_info is None:
                    failed_molecules.append(mol_id)
                    continue
                
                # Extract the key information: structure and properties.
                structures = mol_info.get('molecule_structures') or {}
                properties = mol_info.get('molecule_properties') or {}
                
                # The SMILES string is essential for our GNN model. Skip if it's missing.
                smiles = structures.get('canonical_smiles', '')
                if not smiles:
                    failed_molecules.append(mol_id)
                    continue
                
                # Store the data in a structured dictionary.
                mol_data = {
                    'molecule_chembl_id': mol_id,
                    'canonical_smiles': smiles,
                    'molecular_weight': properties.get('mw_freebase', np.nan),
                    'alogp': properties.get('alogp', np.nan),
                    'hbd': properties.get('hbd', np.nan),
                    'hba': properties.get('hba', np.nan),
                    'psa': properties.get('psa', np.nan),
                    'rtb': properties.get('rtb', np.nan),
                    'compound_name': mol_info.get('pref_name', ''),
                    'max_phase': mol_info.get('max_phase', 0), # Clinical development phase
                    'molecule_type': mol_info.get('molecule_type', ''),
                }
                
                batch_data.append(mol_data)
                
            except Exception as e:
                print(f"   ⚠️ Error with {mol_id}: {str(e)[:50]}...")
                failed_molecules.append(mol_id)
                continue
        
        all_molecule_data.extend(batch_data)
        
        # Save intermediate results for molecule downloads as well.
        if len(all_molecule_data) % 500 == 0:
            temp_mol_df = pd.DataFrame(all_molecule_data)
            temp_mol_df.to_csv(f'temp_molecules_{len(all_molecule_data)}.csv', index=False)
        
        time.sleep(1) # Pause for 1 second between batches.
    
    print(f"✅ Successfully processed {len(all_molecule_data)} molecules")
    print(f"⚠️ Failed to process {len(failed_molecules)} molecules")
    
    return pd.DataFrame(all_molecule_data)

def create_final_large_dataset():
    """
    Orchestrates the creation of the final, large-scale training dataset by calling the
    download, processing, and merging functions.

    Returns:
        pd.DataFrame: The final, analysis-ready dataset.
    """
    
    print("\n📊 CREATING FINAL LARGE DATASET")
    print("=" * 35)
    
    # If we have already downloaded the data, we can load it from the intermediate file
    # to save time on subsequent runs.
    if os.path.exists('all_egfr_activities.csv'):
        print("📂 Loading existing activity data...")
        df_activities = pd.read_csv('all_egfr_activities.csv')
    else:
        print("📥 Downloading new activity data...")
        df_activities = download_large_egfr_dataset()
    
    df_processed = process_and_clean_large_dataset(df_activities)
    
    # Similarly, check for existing molecule data.
    molecule_files = [f for f in os.listdir('.') if f.startswith('temp_molecules_')]
    if molecule_files:
        print("📂 Loading existing molecule data...")
        latest_mol_file = sorted(molecule_files)[-1]
        df_molecules = pd.read_csv(latest_mol_file)
    else:
        print("🧬 Downloading new molecule data...")
        df_molecules = download_molecular_structures_batch(df_processed)
    
    # Merge the activity data with the molecular structure data.
    print("🔗 Merging activity and molecular data...")
    df_final = df_processed.merge(df_molecules, on='molecule_chembl_id', how='inner')
    
    # Apply a final set of quality filters based on Lipinski's Rule of Five.
    print("🔍 Applying final quality filters...")
    df_final = df_final[
        (df_final['molecular_weight'] <= 500) &
        (df_final['alogp'] <= 5) &
        (df_final['hbd'] <= 5) &
        (df_final['hba'] <= 10)
    ].copy()
    
    # Remove molecules with excessively long SMILES strings, which can indicate errors or be difficult to model.
    df_final = df_final[df_final['canonical_smiles'].str.len() <= 200].copy()
    
    # Create the final target columns for the machine learning model.
    # We use the base-10 logarithm of the activity value, which is a standard practice
    # as it compresses the wide range of activity values into a more manageable scale.
    df_final['log_activity'] = np.log10(df_final['activity_value_nm'])
    # We also create a binary classification label: Active (1) or Inactive (0).
    # A common threshold for activity is 1000 nM (1 μM).
    df_final['activity_class'] = (df_final['activity_value_nm'] <= 1000).astype(int)
    # Create descriptive categories for easier analysis.
    df_final['activity_category'] = df_final['activity_value_nm'].apply(
        lambda x: 'Very Active' if x <= 10 else
                 'Active' if x <= 100 else
                 'Moderately Active' if x <= 1000 else
                 'Weakly Active' if x <= 10000 else
                 'Inactive'
    )
    
    # Save the final dataset to a uniquely named file that includes the number of compounds.
    final_filename = f'egfr_dataset_{len(df_final)}_compounds.csv'
    df_final.to_csv(final_filename, index=False)
    
    print(f"💾 Saved final dataset: {final_filename}")
    print(f"📊 Final dataset statistics:")
    print(f"   Total compounds: {len(df_final):,}")
    print(f"   Unique molecules: {df_final['molecule_chembl_id'].nunique():,}")
    print(f"   Activity range: {df_final['activity_value_nm'].min():.1f} - {df_final['activity_value_nm'].max():.1f} nM")
    print(f"   Active compounds (≤1000 nM): {(df_final['activity_class'] == 1).sum():,} ({(df_final['activity_class'] == 1).mean()*100:.1f}%)")
    
    print(f"\n📈 Activity distribution:")
    for category in df_final['activity_category'].value_counts().index:
        count = (df_final['activity_category'] == category).sum()
        pct = count / len(df_final) * 100
        print(f"   {category}: {count:,} ({pct:.1f}%)")
    
    # Clean up the temporary files that were created during the download process.
    print("\n🧹 Cleaning up temporary files...")
    temp_files = [f for f in os.listdir('.') if f.startswith('temp_')]
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            print(f"   Removed: {temp_file}")
        except:
            pass
    
    return df_final

def validate_large_dataset(df):
    """
    Performs a series of validation checks on the final dataset to ensure it is
    of high quality and ready for machine learning.

    Args:
        df (pd.DataFrame): The final dataset to validate.
    """
    
    print(f"\n✅ LARGE DATASET VALIDATION")
    print("=" * 30)
    
    # Basic statistics
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Data quality checks
    print(f"\n🔍 Data quality:")
    print(f"   Valid SMILES: {df['canonical_smiles'].notna().sum():,}/{len(df)} ({df['canonical_smiles'].notna().mean()*100:.1f}%)")
    print(f"   Valid activities: {df['activity_value_nm'].notna().sum():,}/{len(df)} ({df['activity_value_nm'].notna().mean()*100:.1f}%)")
    print(f"   Valid molecular weights: {df['molecular_weight'].notna().sum():,}/{len(df)}")
    
    # Chemical diversity checks
    print(f"\n🧪 Chemical diversity:")
    print(f"   Unique SMILES: {df['canonical_smiles'].nunique():,}")
    print(f"   Average SMILES length: {df['canonical_smiles'].str.len().mean():.1f}")
    print(f"   MW range: {df['molecular_weight'].min():.1f} - {df['molecular_weight'].max():.1f} Da")
    print(f"   LogP range: {df['alogp'].min():.1f} - {df['alogp'].max():.1f}")
    
    # Activity distribution checks
    print(f"\n📊 Activity distribution:")
    print(f"   Activity range: {df['activity_value_nm'].min():.2e} - {df['activity_value_nm'].max():.2e} nM")
    print(f"   Log activity range: {df['log_activity'].min():.1f} - {df['log_activity'].max():.1f}")
    print(f"   Active/Inactive ratio: {df['activity_class'].mean():.3f}")
    
    # Check if known, potent EGFR inhibitors are present in our dataset, which serves as a good sanity check.
    known_drugs = ['GEFITINIB', 'ERLOTINIB', 'LAPATINIB', 'AFATINIB', 'OSIMERTINIB']
    found_drugs = []
    
    print(f"\n💊 Known EGFR drugs:")
    for drug in known_drugs:
        matches = df[df['compound_name'].str.contains(drug, case=False, na=False)]
        if len(matches) > 0:
            found_drugs.append(drug)
            min_activity = matches['activity_value_nm'].min()
            print(f"   ✅ {drug}: {len(matches)} records, best activity: {min_activity:.1f} nM")
    
    # A final summary of the dataset's readiness for machine learning.
    print(f"\n🎯 Dataset readiness for ML:")
    readiness_checks = {
        "Sufficient size (>1000)": len(df) > 1000,
        "High data quality (>95%)": df['canonical_smiles'].notna().mean() > 0.95,
        "Good activity range (>4 orders)": (df['activity_value_nm'].max() / df['activity_value_nm'].min()) > 10000,
        "Balanced classes (20-80% active)": 0.2 <= df['activity_class'].mean() <= 0.8,
        "Chemical diversity (unique SMILES)": df['canonical_smiles'].nunique() / len(df) > 0.8,
        "Known drugs present": len(found_drugs) >= 3
    }
    
    passed_checks = 0
    for check, passed in readiness_checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check}")
        if passed:
            passed_checks += 1
    
    score = passed_checks / len(readiness_checks) * 100
    print(f"\n📊 Overall readiness score: {score:.0f}%")
    
    if score >= 85:
        print("🎉 EXCELLENT! Dataset is ready for high-performance GNN training!")
    elif score >= 70:
        print("✅ GOOD! Dataset suitable for GNN training with good results expected.")
    else:
        print("⚠️ NEEDS IMPROVEMENT: Consider additional data collection or cleaning.")

def main():
    """
    Main execution function that runs the entire dataset creation pipeline.
    """
    
    print("🚀 LARGE-SCALE EGFR DATASET CREATION")
    print("=" * 50)
    print("Target: 10,000+ compounds for enhanced GNN training")
    print("This process may take 1-2 hours for comprehensive data collection\n")
    
    # Create the dataset.
    df_large = create_final_large_dataset()
    
    # Validate the created dataset.
    validate_large_dataset(df_large)
    
    print(f"\n🎉 LARGE DATASET CREATION COMPLETE!")
    print(f"Ready for enhanced GNN training with {len(df_large):,} compounds!")

# Standard Python entry point.
if __name__ == "__main__":
    main()
