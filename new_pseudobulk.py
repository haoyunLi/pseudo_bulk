import pandas as pd 
import numpy as np 
import logging 
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def generate_random_celltype_percentages(n_cell_types, n_donors):
    """
    Generate random cell type percentages for each donor that sum to 1.
    
    Args:
        n_cell_types: Number of cell types
        n_donors: Number of donors
        
    Returns:
        Array of shape (n_donors, n_cell_types) with random percentages
    """
    # Generate random numbers
    random_percentages = np.random.dirichlet(np.ones(n_cell_types), size=n_donors)
    return random_percentages

def save_random_data(percentages_df, cell_types, donors, output_dir):
    """
    Save random data and metadata to JSON file.
    
    Args:
        percentages_df: DataFrame with random percentages
        cell_types: List of cell type names
        donors: List of donor IDs
        output_dir: Directory to save the JSON file
    """
    # Create metadata dictionary
    metadata = {
        'random_seed': RANDOM_SEED,
        'n_cell_types': len(cell_types),
        'n_donors': len(donors),
        'cell_types': cell_types.tolist(),
        'donors': donors.tolist(),
        'percentages': {
            donor: {
                cell_type: float(percentages_df.loc[donor, cell_type])
                for cell_type in cell_types
            }
            for donor in donors
        }
    }
    
    # Save to JSON file
    json_path = os.path.join(output_dir, 'random_data_metadata.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logging.info(f"Random data metadata saved to {json_path}")

def create_new_pseudobulk(celltype_matrix, cell_types, donors, random_percentages, genes):
    """
    Create pseudobulk data using random cell type percentages.
    
    Args:
        celltype_matrix: 3D matrix of shape (n_cell_types, n_donors, n_genes)
        cell_types: List of cell type names
        donors: List of donor IDs
        random_percentages: Array of shape (n_donors, n_cell_types) with random percentages
        genes: List of gene names
        
    Returns:
        DataFrame with pseudobulk expression data
    """
    n_cell_types, n_donors, n_genes = celltype_matrix.shape
    
    # Initialize pseudobulk matrix
    pseudobulk_matrix = np.zeros((n_donors, n_genes))
    
    # For each donor
    for d in range(n_donors):
        # For each cell type
        for ct in range(n_cell_types):
            # Get the cell type expression and multiply by its percentage
            celltype_expr = celltype_matrix[ct, d, :]
            weighted_expr = celltype_expr * random_percentages[d, ct]
            pseudobulk_matrix[d, :] += weighted_expr
    
    # Create DataFrame
    pseudobulk_df = pd.DataFrame(
        pseudobulk_matrix,
        index=donors,
        columns=genes
    )
    
    # Save the random percentages
    percentages_df = pd.DataFrame(
        random_percentages,
        index=donors,
        columns=cell_types
    )
    percentages_df.to_csv('data/new_celltype_percentages.csv')
    
    return pseudobulk_df

def main():
    # Create output directory
    os.makedirs('data', exist_ok=True)
    
    # Load existing celltype matrix data
    logging.info("Loading existing celltype matrix data...")
    celltype_matrix = np.load('data/celltype_specific_3d_3d_matrix.npy')
    cell_types = pd.read_csv('data/celltype_specific_3d_cell_types.csv', header=None)[0].tolist()
    donors = pd.read_csv('data/celltype_specific_3d_donors.csv', header=None)[0].tolist()
    genes = pd.read_csv('data/celltype_specific_3d_genes.csv', header=None)[0].tolist()
    
    logging.info(f"Loaded matrix with shape: {celltype_matrix.shape}")
    logging.info(f"Number of cell types: {len(cell_types)}")
    logging.info(f"Number of donors: {len(donors)}")
    logging.info(f"Number of genes: {len(genes)}")
    
    # Generate random percentages
    logging.info("Generating random cell type percentages...")
    random_percentages = generate_random_celltype_percentages(
        len(cell_types),
        len(donors)
    )
    
    # Create new pseudobulk
    logging.info("Creating new pseudobulk data...")
    pseudobulk_df = create_new_pseudobulk(
        celltype_matrix,
        cell_types,
        donors,
        random_percentages,
        genes
    )
    
    # Save results
    logging.info("Saving results...")
    pseudobulk_df.to_csv('data/new_pseudobulk_expression.csv')
    
    # Save random data metadata
    percentages_df = pd.DataFrame(
        random_percentages,
        index=donors,
        columns=cell_types
    )
    save_random_data(
        percentages_df,
        np.array(cell_types),
        np.array(donors),
        'data'
    )
    
    # Log some statistics
    logging.info(f"New pseudobulk shape: {pseudobulk_df.shape}")
    logging.info("Random percentages saved to data/new_celltype_percentages.csv")
    logging.info(f"Random seed used: {RANDOM_SEED}")
    
    # Print some example percentages
    logging.info("\nExample random percentages for first donor:")
    print(percentages_df.iloc[0].to_string())

if __name__ == "__main__":
    main()

