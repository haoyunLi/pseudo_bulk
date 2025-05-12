import pandas as pd
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def split_data(input_file, test_size=50, random_state=42):
    """Split data into training and testing sets."""
    logging.info(f"Reading data from {input_file}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file, index_col=0)
    
    # Get total number of samples
    total_samples = len(df)
    logging.info(f"Total number of samples: {total_samples}")
    
    # Verify original data format
    original_cols = len(df.columns)
    logging.info(f"Original number of columns: {original_cols}")
    
    # Randomly select test samples
    np.random.seed(random_state)
    test_indices = np.random.choice(total_samples, size=test_size, replace=False)
    
    # Split the data
    test_df = df.iloc[test_indices].copy()  # Use copy to ensure data integrity
    train_df = df.drop(df.index[test_indices]).copy()
    
    # Verify split data format
    test_cols = len(test_df.columns)
    train_cols = len(train_df.columns)
    
    if test_cols != original_cols or train_cols != original_cols:
        raise ValueError(f"Column count mismatch! Original: {original_cols}, Test: {test_cols}, Train: {train_cols}")
    
    # Create output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save the split data
    test_file = 'data/test_pseudobulk.csv'
    train_file = 'data/train_pseudobulk.csv'
    
    # Save with explicit index=True to ensure format consistency
    test_df.to_csv(test_file, index=True)
    train_df.to_csv(train_file, index=True)
    
    # Verify saved files
    test_loaded = pd.read_csv(test_file, index_col=0)
    if len(test_loaded.columns) != original_cols:
        raise ValueError(f"Saved test file has incorrect number of columns: {len(test_loaded.columns)} vs {original_cols}")
    
    logging.info(f"Test data ({len(test_df)} samples) saved to {test_file}")
    logging.info(f"Training data ({len(train_df)} samples) saved to {train_file}")
    logging.info(f"Column count verified: {original_cols} columns maintained in all files")
    
    return test_file, train_file

def main():
    # Split the data
    test_file, train_file = split_data(
        'data/processed_pseudobulk_expression_W.csv',
        test_size=50
    )
    
    logging.info("Data splitting completed successfully!")

if __name__ == "__main__":
    main() 