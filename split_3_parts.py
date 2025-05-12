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

def split_into_three_parts(input_file, output_prefix, random_state=42):
    """Split data into three equal parts."""
    logging.info(f"Reading data from {input_file}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file, index_col=0)
    
    # Get total number of samples
    total_samples = len(df)
    part_size = total_samples // 3
    logging.info(f"Total samples: {total_samples}, Part size: {part_size}")
    
    # Split into three parts sequentially
    part1_indices = np.arange(0, part_size)
    part2_indices = np.arange(part_size, 2*part_size)
    part3_indices = np.arange(2*part_size, total_samples)
    
    # Create the three parts
    part1_df = df.iloc[part1_indices]
    part2_df = df.iloc[part2_indices]
    part3_df = df.iloc[part3_indices]
    
    # Save the parts
    part1_file = f'data/{output_prefix}_part1.csv'
    part2_file = f'data/{output_prefix}_part2.csv'
    part3_file = f'data/{output_prefix}_part3.csv'
    
    part1_df.to_csv(part1_file)
    part2_df.to_csv(part2_file)
    part3_df.to_csv(part3_file)
    
    logging.info(f"Part 1 ({len(part1_df)} samples) saved to {part1_file}")
    logging.info(f"Part 2 ({len(part2_df)} samples) saved to {part2_file}")
    logging.info(f"Part 3 ({len(part3_df)} samples) saved to {part3_file}")
    
    return part1_file, part2_file, part3_file

def main():
    # Split pseudobulk data
    logging.info("Splitting pseudobulk data...")
    split_into_three_parts(
        'data/train_pseudobulk_randomized.csv',
        'pseudobulk_randomized'
    )
    """
    # Split celltype specific data
    logging.info("Splitting celltype specific data...")
    split_into_three_parts(
        'data/celltype_specific_2d_matrix_sorted.csv',
        'celltype'
    )
    """
    logging.info("All data has been split into three parts successfully!")

if __name__ == "__main__":
    main() 