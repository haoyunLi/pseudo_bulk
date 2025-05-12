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

def randomize_pseudobulk(input_file, output_file, random_state=42):
    """Randomly reorder the pseudobulk data and save it."""
    logging.info(f"Reading data from {input_file}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file, index_col=0)
    
    # Get total number of samples
    total_samples = len(df)
    logging.info(f"Total samples: {total_samples}")
    
    # Create random permutation of indices
    np.random.seed(random_state)
    random_indices = np.random.permutation(total_samples)
    
    # Reorder the dataframe
    randomized_df = df.iloc[random_indices]
    
    # Save the randomized data
    randomized_df.to_csv(output_file)
    logging.info(f"Randomized data saved to {output_file}")

def main():
    input_file = 'data/train_pseudobulk_sorted.csv'
    output_file = 'data/train_pseudobulk_randomized.csv'
    
    randomize_pseudobulk(input_file, output_file)
    logging.info("Data has been randomly reordered successfully!")

if __name__ == "__main__":
    main() 