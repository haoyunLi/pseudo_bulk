import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def extract_donor_id(index):
    """Extract donor ID from the index string (format: 'celltype|donorID')."""
    return index.split('|')[1] if '|' in index else index

def sort_data_by_donor(input_file, output_file):
    """Sort data by donor ID and save to a new file."""
    logging.info(f"Processing {input_file}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file, index_col=0)
    
    # Extract donor IDs and create a temporary column for sorting
    df['donor_id'] = df.index.map(extract_donor_id)
    
    # Sort by donor ID
    df_sorted = df.sort_values('donor_id')
    
    # Remove the temporary column
    df_sorted = df_sorted.drop('donor_id', axis=1)
    
    # Save the sorted data
    df_sorted.to_csv(output_file)
    logging.info(f"Sorted data saved to {output_file}")

def main():
    # Sort celltype specific data
    sort_data_by_donor(
        'data/celltype_specific_2d_matrix.csv',
        'data/celltype_specific_2d_matrix_sorted.csv'
    )
    
    # Sort pseudobulk data
    sort_data_by_donor(
        'data/train_pseudobulk.csv',
        'data/train_pseudobulk_sorted.csv'
    )
    
    logging.info("All files have been sorted successfully!")

if __name__ == "__main__":
    main() 