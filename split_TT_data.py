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
    
    # Randomly select test samples
    np.random.seed(random_state)
    test_indices = np.random.choice(total_samples, size=test_size, replace=False)
    
    # Split the data
    test_df = df.iloc[test_indices]
    train_df = df.drop(df.index[test_indices])
    
    # Create output directory if it doesn't exist
    os.makedirs('data/split', exist_ok=True)
    
    # Save the split data
    test_file = 'data/split/test_pseudobulk.csv'
    train_file = 'data/split/train_pseudobulk.csv'
    
    test_df.to_csv(test_file)
    train_df.to_csv(train_file)
    
    logging.info(f"Test data ({len(test_df)} samples) saved to {test_file}")
    logging.info(f"Training data ({len(train_df)} samples) saved to {train_file}")
    
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