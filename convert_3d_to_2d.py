import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def load_3d_matrix_data(matrix_path: str, cell_types_path: str, donors_path: str, genes_path: str) -> Dict:
    """
    Load the 3D matrix and associated metadata from files.
    
    @param matrix_path: Path to the 3D matrix .npy file
    @param cell_types_path: Path to the cell types CSV file
    @param donors_path: Path to the donors CSV file
    @param genes_path: Path to the genes CSV file
    @return: Dictionary containing 3D matrix and metadata
    """
    # Load 3D matrix
    matrix = np.load(matrix_path)
    
    # Load metadata
    cell_types = pd.read_csv(cell_types_path, header=None)[0].tolist()
    donors = pd.read_csv(donors_path, header=None)[0].tolist()
    genes = pd.read_csv(genes_path, header=None)[0].tolist()
    
    return {
        'matrix': matrix,
        'cell_types': cell_types,
        'donors': donors,
        'genes': genes
    }

def load_common_genes(common_genes_path: str) -> List[str]:
    """
    Load common gene IDs from a text file.
    
    @param common_genes_path: Path to the common gene IDs file
    @return: List of common gene IDs
    """
    with open(common_genes_path, "r") as f:
        return [line.strip() for line in f.readlines()]

def verify_gene_ids_and_order(processed_data: pd.DataFrame, common_genes: List[str]) -> Tuple[bool, bool]:
    """
    Verify if gene IDs and their order match between processed data and common genes list.
    
    @param processed_data: Processed DataFrame with gene expression data
    @param common_genes: List of common gene IDs
    @return: Tuple of (gene_ids_match, gene_order_match)
    """
    # Get gene columns from processed data
    processed_genes = processed_data.columns.tolist()
    
    # Check if gene IDs are identical
    gene_ids_match = set(common_genes) == set(processed_genes)
    if not gene_ids_match:
        print("Warning: Gene IDs are not identical between common gene list and processed data")
        print(f"Genes in common list but not in processed data: {set(common_genes) - set(processed_genes)}")
        print(f"Genes in processed data but not in common list: {set(processed_genes) - set(common_genes)}")
    else:
        print("Gene IDs are identical between common gene list and processed data")
    
    # Check if order is identical
    gene_order_match = common_genes == processed_genes
    if not gene_order_match:
        print("Warning: Gene order is not identical between common gene list and processed data")
    else:
        print("Gene order is identical between common gene list and processed data")
    
    return gene_ids_match, gene_order_match

def convert_3d_to_2d(result: Dict, common_genes_path: str) -> pd.DataFrame:
    """
    Convert 3D matrix to 2D matrix and verify against common gene list.
    
    @param result: Dictionary containing 3D matrix and metadata
    @param common_genes_path: Path to the common gene IDs file
    @return: DataFrame with processed 2D matrix
    """
    # Load common genes
    common_genes = load_common_genes(common_genes_path)
    
    # Get current genes from 3D matrix
    current_genes = result['genes']
    
    # Check for genes in 3D matrix but not in common genes
    missing_in_common = set(current_genes) - set(common_genes)
    if missing_in_common:
        print(f"Warning: {len(missing_in_common)} genes in 3D matrix are not in common genes list")
    
    # Check for genes in common list but not in 3D matrix
    missing_in_3d = set(common_genes) - set(current_genes)
    if missing_in_3d:
        print(f"Note: {len(missing_in_3d)} genes in common list are not in 3D matrix")
        print("These genes will be added with zero expression values")
    
    # Reshape 3D matrix to 2D (cell_types x donors, genes)
    n_cell_types = len(result['cell_types'])
    n_donors = len(result['donors'])
    expression_2d = result['matrix'].reshape(n_cell_types * n_donors, -1)
    
    # Create row labels (cell_type|donor)
    row_labels = []
    for cell_type in result['cell_types']:
        for donor in result['donors']:
            row_labels.append(f"{cell_type}|{donor}")
    
    # Create DataFrame with current genes
    df = pd.DataFrame(expression_2d, index=row_labels, columns=current_genes)
    
    # Create a dictionary to store gene columns
    gene_columns = {}
    
    # Add genes in the order of common_genes
    for gene in common_genes:
        if gene in current_genes:
            # If gene exists in 3D matrix, use its expression values
            gene_columns[gene] = df[gene]
        else:
            # If gene is missing, add it with zero expression
            gene_columns[gene] = pd.Series(0, index=df.index)
    
    # Create new DataFrame with genes in common_genes order
    processed_df = pd.DataFrame(gene_columns, index=df.index)
    
    # Verify gene IDs and order
    verify_gene_ids_and_order(processed_df, common_genes)
    
    return processed_df

def save_2d_matrix(processed_df: pd.DataFrame, output_path: str) -> None:
    """
    Save the processed 2D matrix to a CSV file.
    
    @param processed_df: Processed DataFrame to save
    @param output_path: Path where to save the processed data
    """
    processed_df.to_csv(output_path)
    print(f"Saved processed 2D matrix to {output_path}")

if __name__ == "__main__":
    # File paths
    matrix_path = 'data/celltype_specific_3d_3d_matrix.npy'
    cell_types_path = 'data/celltype_specific_3d_cell_types.csv'
    donors_path = 'data/celltype_specific_3d_donors.csv'
    genes_path = 'data/celltype_specific_3d_genes.csv'
    common_genes_path = 'data/common_gene_id.txt'
    output_path = 'data/celltype_specific_2d_matrix.csv'
    
    # Load 3D matrix data
    print("Loading 3D matrix data...")
    result = load_3d_matrix_data(matrix_path, cell_types_path, donors_path, genes_path)
    
    # Convert to 2D and verify against common genes
    print("Converting to 2D matrix and verifying against common genes...")
    processed_df = convert_3d_to_2d(result, common_genes_path)
    
    # Save processed 2D matrix
    save_2d_matrix(processed_df, output_path) 