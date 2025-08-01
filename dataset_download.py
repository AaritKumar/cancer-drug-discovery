import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
import time
import os
from datetime import datetime

def download_large_egfr_dataset():
    """Download comprehensive EGFR bioactivity dataset from ChEMBL"""
    
    print("🚀 LARGE-SCALE EGFR DATASET COLLECTION")
    print("=" * 45)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize ChEMBL clients
    activity = new_client.activity
    molecule = new_client.molecule
    target = new_client.target
    
    # Verify EGFR target
    print("🎯 Verifying EGFR target...")
    egfr_target = target.get('CHEMBL203')
    print(f"Target: {egfr_target['pref_name']}")
    
    # Download comprehensive bioactivity data with relaxed filters
    print("📥 Downloading comprehensive EGFR bioactivity data...")
    print("This will take 30-60 minutes for large dataset...")
    
    # Broader activity filters for more data
    activities = activity.filter(
        target_chembl_id='CHEMBL203',  # EGFR
        standard_type__in=['IC50', 'EC50', 'Ki', 'Kd', 'GI50'],  # More activity types
        standard_relation__in=['=', '<', '<=', '>', '>='],  # Include approximate values
        standard_units__in=['nM', 'uM', 'pM'],  # Multiple units
        confidence_score__gte=5,  # Lower confidence threshold for more data
        pchembl_value__isnull=False  # Must have pChEMBL value
    )
    
    # Convert to list with progress tracking
    print("🔄 Processing activities (this takes time)...")
    activities_list = []
    count = 0
    
    for activity_data in activities:
        activities_list.append(activity_data)
        count += 1
        
        if count % 500 == 0:
            print(f"   Downloaded {count} records...")
            
        # Save intermediate results every 2000 records
        if count % 2000 == 0:
            temp_df = pd.DataFrame(activities_list)
            temp_df.to_csv(f'temp_activities_{count}.csv', index=False)
            print(f"   Saved intermediate file: temp_activities_{count}.csv")
        
        # Don't overwhelm the server
        if count % 1000 == 0:
            time.sleep(2)
    
    print(f"✅ Downloaded {len(activities_list)} activity records")
    
    # Convert to DataFrame and save
    df_activities = pd.DataFrame(activities_list)
    df_activities.to_csv('all_egfr_activities.csv', index=False)
    print(f"💾 Saved all activities to: all_egfr_activities.csv")
    
    return df_activities

def process_and_clean_large_dataset(df_activities):
    """Process and clean the large dataset"""
    
    print("\n🧹 PROCESSING AND CLEANING LARGE DATASET")
    print("=" * 45)
    
    print(f"Starting with {len(df_activities)} raw activity records")
    
    # Data cleaning and standardization
    print("🔧 Cleaning and standardizing data...")
    
    # Convert units to consistent nM
    def convert_to_nm(value, unit):
        if pd.isna(value) or pd.isna(unit):
            return np.nan
        
        unit = str(unit).lower()
        if 'nm' in unit:
            return float(value)
        elif 'um' in unit or 'μm' in unit:
            return float(value) * 1000  # μM to nM
        elif 'pm' in unit:
            return float(value) / 1000  # pM to nM
        else:
            return np.nan
    
    # Apply unit conversion
    df_activities['standard_value_nm'] = df_activities.apply(
        lambda row: convert_to_nm(row['standard_value'], row['standard_units']), axis=1
    )
    
    # Filter for reasonable activity ranges (0.1 nM to 1M nM)
    df_clean = df_activities[
        (df_activities['standard_value_nm'] >= 0.1) & 
        (df_activities['standard_value_nm'] <= 1000000)
    ].copy()
    
    print(f"After unit conversion and range filtering: {len(df_clean)} records")
    
    # Handle relation symbols (convert < and > to approximate values)
    def handle_relations(row):
        value = row['standard_value_nm']
        relation = row.get('standard_relation', '=')
        
        if pd.isna(value):
            return np.nan
        
        if relation == '<':
            return value / 2  # Conservative estimate
        elif relation == '<=':
            return value
        elif relation == '>':
            return value * 2  # Conservative estimate
        elif relation == '>=':
            return value
        else:  # '=' or other
            return value
    
    df_clean['activity_value_nm'] = df_clean.apply(handle_relations, axis=1)
    
    # Remove duplicates based on molecule and keep best measurement
    print("🔄 Removing duplicates and selecting best measurements...")
    
    # Group by molecule and activity type, keep measurement with highest confidence
    df_dedup = df_clean.sort_values(['molecule_chembl_id', 'standard_type', 'confidence_score'], 
                                   ascending=[True, True, False])
    df_dedup = df_dedup.groupby(['molecule_chembl_id', 'standard_type']).first().reset_index()
    
    print(f"After deduplication: {len(df_dedup)} records")
    
    # For molecules with multiple activity types, prioritize IC50 > Ki > EC50 > others
    activity_priority = {'IC50': 1, 'Ki': 2, 'EC50': 3, 'Kd': 4, 'GI50': 5}
    df_dedup['priority'] = df_dedup['standard_type'].map(activity_priority).fillna(10)
    
    df_final = df_dedup.sort_values(['molecule_chembl_id', 'priority']).groupby('molecule_chembl_id').first().reset_index()
    
    print(f"After activity type prioritization: {len(df_final)} unique compounds")
    
    return df_final

def download_molecular_structures_batch(df_final):
    """Download molecular structures in batches for efficiency"""
    
    print("\n🧬 DOWNLOADING MOLECULAR STRUCTURES (BATCH MODE)")
    print("=" * 50)
    
    molecule = new_client.molecule
    unique_molecules = df_final['molecule_chembl_id'].unique()
    total_molecules = len(unique_molecules)
    
    print(f"Processing {total_molecules} unique molecules...")
    
    # Process in batches
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
                
                # Extract molecular information
                structures = mol_info.get('molecule_structures') or {}
                properties = mol_info.get('molecule_properties') or {}
                
                smiles = structures.get('canonical_smiles', '')
                if not smiles:  # Skip molecules without SMILES
                    failed_molecules.append(mol_id)
                    continue
                
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
                    'max_phase': mol_info.get('max_phase', 0),
                    'molecule_type': mol_info.get('molecule_type', ''),
                }
                
                batch_data.append(mol_data)
                
            except Exception as e:
                print(f"   ⚠️ Error with {mol_id}: {str(e)[:50]}...")
                failed_molecules.append(mol_id)
                continue
        
        all_molecule_data.extend(batch_data)
        
        # Save intermediate results
        if len(all_molecule_data) % 500 == 0:
            temp_mol_df = pd.DataFrame(all_molecule_data)
            temp_mol_df.to_csv(f'temp_molecules_{len(all_molecule_data)}.csv', index=False)
        
        # Be nice to the server
        time.sleep(1)
    
    print(f"✅ Successfully processed {len(all_molecule_data)} molecules")
    print(f"⚠️ Failed to process {len(failed_molecules)} molecules")
    
    return pd.DataFrame(all_molecule_data)

def create_final_large_dataset():
    """Create the final large training dataset"""
    
    print("\n📊 CREATING FINAL LARGE DATASET")
    print("=" * 35)
    
    # Check if intermediate files exist
    if os.path.exists('all_egfr_activities.csv'):
        print("📂 Loading existing activity data...")
        df_activities = pd.read_csv('all_egfr_activities.csv')
    else:
        print("📥 Downloading new activity data...")
        df_activities = download_large_egfr_dataset()
    
    # Process and clean
    df_processed = process_and_clean_large_dataset(df_activities)
    
    # Check for existing molecule data
    molecule_files = [f for f in os.listdir('.') if f.startswith('temp_molecules_')]
    if molecule_files:
        print("📂 Loading existing molecule data...")
        latest_mol_file = sorted(molecule_files)[-1]
        df_molecules = pd.read_csv(latest_mol_file)
    else:
        print("🧬 Downloading new molecule data...")
        df_molecules = download_molecular_structures_batch(df_processed)
    
    # Merge datasets
    print("🔗 Merging activity and molecular data...")
    df_final = df_processed.merge(df_molecules, on='molecule_chembl_id', how='inner')
    
    # Final quality filters
    print("🔍 Applying final quality filters...")
    
    # Filter for drug-like molecules (Lipinski's Rule of Five)
    df_final = df_final[
        (df_final['molecular_weight'] <= 500) &
        (df_final['alogp'] <= 5) &
        (df_final['hbd'] <= 5) &
        (df_final['hba'] <= 10)
    ].copy()
    
    # Remove very long SMILES (likely errors or complex molecules)
    df_final = df_final[df_final['canonical_smiles'].str.len() <= 200].copy()
    
    # Add activity classifications
    df_final['log_activity'] = np.log10(df_final['activity_value_nm'])
    df_final['activity_class'] = (df_final['activity_value_nm'] <= 1000).astype(int)
    df_final['activity_category'] = df_final['activity_value_nm'].apply(
        lambda x: 'Very Active' if x <= 10 else
                 'Active' if x <= 100 else
                 'Moderately Active' if x <= 1000 else
                 'Weakly Active' if x <= 10000 else
                 'Inactive'
    )
    
    # Save final dataset
    final_filename = f'egfr_large_dataset_{len(df_final)}_compounds.csv'
    df_final.to_csv(final_filename, index=False)
    
    print(f"💾 Saved final dataset: {final_filename}")
    print(f"📊 Final dataset statistics:")
    print(f"   Total compounds: {len(df_final):,}")
    print(f"   Unique molecules: {df_final['molecule_chembl_id'].nunique():,}")
    print(f"   Activity range: {df_final['activity_value_nm'].min():.1f} - {df_final['activity_value_nm'].max():.1f} nM")
    print(f"   Active compounds (≤1000 nM): {(df_final['activity_class'] == 1).sum():,} ({(df_final['activity_class'] == 1).mean()*100:.1f}%)")
    
    # Activity distribution
    print(f"\n📈 Activity distribution:")
    for category in df_final['activity_category'].value_counts().index:
        count = (df_final['activity_category'] == category).sum()
        pct = count / len(df_final) * 100
        print(f"   {category}: {count:,} ({pct:.1f}%)")
    
    # Clean up temporary files
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
    """Comprehensive validation of the large dataset"""
    
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
    
    # Chemical diversity
    print(f"\n🧪 Chemical diversity:")
    print(f"   Unique SMILES: {df['canonical_smiles'].nunique():,}")
    print(f"   Average SMILES length: {df['canonical_smiles'].str.len().mean():.1f}")
    print(f"   MW range: {df['molecular_weight'].min():.1f} - {df['molecular_weight'].max():.1f} Da")
    print(f"   LogP range: {df['alogp'].min():.1f} - {df['alogp'].max():.1f}")
    
    # Activity distribution
    print(f"\n📊 Activity distribution:")
    print(f"   Activity range: {df['activity_value_nm'].min():.2e} - {df['activity_value_nm'].max():.2e} nM")
    print(f"   Log activity range: {df['log_activity'].min():.1f} - {df['log_activity'].max():.1f}")
    print(f"   Active/Inactive ratio: {df['activity_class'].mean():.3f}")
    
    # Known drugs check
    known_drugs = ['GEFITINIB', 'ERLOTINIB', 'LAPATINIB', 'AFATINIB', 'OSIMERTINIB']
    found_drugs = []
    
    print(f"\n💊 Known EGFR drugs:")
    for drug in known_drugs:
        matches = df[df['compound_name'].str.contains(drug, case=False, na=False)]
        if len(matches) > 0:
            found_drugs.append(drug)
            min_activity = matches['activity_value_nm'].min()
            print(f"   ✅ {drug}: {len(matches)} records, best activity: {min_activity:.1f} nM")
    
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
    """Main execution function"""
    
    print("🚀 LARGE-SCALE EGFR DATASET CREATION")
    print("=" * 50)
    print("Target: 10,000+ compounds for enhanced GNN training")
    print("This process may take 1-2 hours for comprehensive data collection\n")
    
    # Create large dataset
    df_large = create_final_large_dataset()
    
    # Validate dataset
    validate_large_dataset(df_large)
    
    print(f"\n🎉 LARGE DATASET CREATION COMPLETE!")
    print(f"Ready for enhanced GNN training with {len(df_large):,} compounds!")

if __name__ == "__main__":
    main()