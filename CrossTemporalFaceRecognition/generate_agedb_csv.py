#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate CSV files for AgeDB human dataset
Similar format to orangutan dataset
"""

import os
import re
import pandas as pd
from pathlib import Path

# Paths
DATA_ROOT = r'D:\Pycharmxm\pytorch_main\Project_KSJ\data\82bd6-main\AgeDB'
OUTPUT_DIR = r'D:\Pycharmxm\pytorch_main\Project_KSJ\CrossTemporalFaceRecognition\splits'
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2


def parse_filename(filename):
    """Parse AgeDB filename: {id}_{name}_{age}_{gender}.jpg"""
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('_')
    
    if len(parts) >= 4:
        # Format: id_name_age_gender.jpg
        identity_id = parts[1]  # name
        age = int(parts[2])
        gender = 'Male' if parts[3] == 'm' else 'Female'
        return identity_id, age, age, gender
    return None, None, None, None


def get_age_group(age):
    """Categorize age into age groups"""
    if age < 0:
        return 'Unknown'
    elif age <= 2:
        return 'Infant'
    elif age <= 12:
        return 'Child'
    elif age <= 17:
        return 'Juvenile'
    elif age <= 35:
        return 'Adult'
    elif age <= 55:
        return 'Middle'
    elif age <= 70:
        return 'Senior'
    else:
        return 'Elderly'


def main():
    print("Scanning AgeDB dataset...")
    
    # Scan all images
    data = []
    image_dir = Path(DATA_ROOT)
    
    for img_path in image_dir.glob('*.jpg'):
        filename = img_path.name
        identity_id, year, age, gender = parse_filename(filename)
        
        if identity_id is not None:
            age_group = get_age_group(age)
            data.append({
                'filename': filename,
                'identity_id': identity_id,
                'year': year,  # Using age as year for consistency
                'age': age,
                'age_group': age_group,
                'gender': gender
            })
    
    df = pd.DataFrame(data)
    print(f"Total images: {len(df)}")
    print(f"Unique identities: {df['identity_id'].nunique()}")
    print(f"Age range: {df['age'].min()} - {df['age'].max()}")
    print(f"Age group distribution:\n{df['age_group'].value_counts()}")
    
    # Check cross-age identities
    identity_years = df.groupby('identity_id')['year'].nunique()
    multi_year_ids = identity_years[identity_years > 1]
    print(f"\nIdentities with multiple ages: {len(multi_year_ids)} / {df['identity_id'].nunique()}")
    
    # Split by identity (not by random) to avoid data leakage
    # For each identity, split images into train/val/test
    identities = df['identity_id'].unique()
    n_identities = len(identities)
    
    # Shuffle identities
    import random
    random.seed(42)
    shuffled_ids = list(identities)
    random.shuffle(shuffled_ids)
    
    # Split identities
    n_train = int(n_identities * TRAIN_RATIO)
    n_val = int(n_identities * VAL_RATIO)
    
    train_ids = set(shuffled_ids[:n_train])
    val_ids = set(shuffled_ids[n_train:n_train + n_val])
    test_ids = set(shuffled_ids[n_train + n_val:])
    
    print(f"\nIdentity split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    
    # Create splits
    train_df = df[df['identity_id'].isin(train_ids)]
    val_df = df[df['identity_id'].isin(val_ids)]
    test_df = df[df['identity_id'].isin(test_ids)]
    
    print(f"Sample split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Save CSVs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'agedb_train.csv'), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, 'agedb_val.csv'), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'agedb_test.csv'), index=False)
    
    print(f"\nCSV files saved to {OUTPUT_DIR}")
    print("- agedb_train.csv")
    print("- agedb_val.csv")
    print("- agedb_test.csv")
    
    # Print some examples
    print("\n=== Sample data ===")
    print(train_df.head(10).to_string())


if __name__ == '__main__':
    main()
