import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
from pseudo_bulk import load_data, get_summary

def create_celltype_specific_3d_matrix(adata):
    """
    Create a 3D matrix of expression data organized by cell types, donors, and genes.
    
    @param adata: AnnData object containing single-cell data
    @return: Dictionary containing:
        - 3D matrix (cell_types x donors x genes)
        - cell type labels
        - donor labels
        - gene labels
    """
    
    # Get unique cell types and donors
    cell_types = sorted(adata.obs['cell_type'].unique())
    donors = sorted(adata.obs['donor_id'].unique())
    genes = adata.var_names
    
    # Initialize 3D matrix
    n_cell_types = len(cell_types)
    n_donors = len(donors)
    n_genes = len(genes)
    
    # Create 3D matrix (cell_types x donors x genes)
    expression_3d = np.zeros((n_cell_types, n_donors, n_genes))
    
    # Fill the 3D matrix
    for i, cell_type in enumerate(cell_types):
        for j, donor in enumerate(donors):
            # Get cells matching this cell type and donor
            mask = (adata.obs['cell_type'] == cell_type) & (adata.obs['donor_id'] == donor)
            if mask.any():
                # Get expression data and average across cells
                cell_expr = adata[mask].X
                if isinstance(cell_expr, np.matrix):
                    cell_expr = np.array(cell_expr)
                # Take mean across cells
                expression_3d[i, j, :] = cell_expr.mean(axis=0)
    
    # Create result dictionary
    result = {
        'matrix': expression_3d,
        'cell_types': cell_types,
        'donors': donors,
        'genes': genes
    }
    return result

def save_3d_matrix(result, output_prefix):
    """
    Save the 3D matrix and associated metadata to files
    
    @param result: Dictionary containing 3D matrix and metadata
    @param output_prefix: Prefix for output files
    """
    print(f"Saving 3D matrix data to {output_prefix}...")
    
    # Save 3D matrix as numpy array
    np.save(f'data/{output_prefix}_3d_matrix.npy', result['matrix'])
    
    # Save metadata
    pd.DataFrame(result['cell_types']).to_csv(f'data/{output_prefix}_cell_types.csv', index=False, header=False)
    pd.DataFrame(result['donors']).to_csv(f'data/{output_prefix}_donors.csv', index=False, header=False)
    pd.DataFrame(result['genes']).to_csv(f'data/{output_prefix}_genes.csv', index=False, header=False)

if __name__ == '__main__':
    # Load data
    adata = load_data('data/08984b3c-3189-4732-be22-62f1fe8f15a4.h5ad')
    
    # Get summary statistics
    stats = get_summary(adata)
    print(f"Found {stats['n_donors']} donors and {stats['n_cell_types']} cell types")
    
    # Create 3D matrix
    result = create_celltype_specific_3d_matrix(adata)
    
    # Save results
    save_3d_matrix(result, 'celltype_specific_3d')
    
    # Print matrix shape
    print(f"3D matrix shape: {result['matrix'].shape}")
    print(f"Number of cell types: {len(result['cell_types'])}")
    print(f"Number of donors: {len(result['donors'])}")
    print(f"Number of genes: {len(result['genes'])}") 