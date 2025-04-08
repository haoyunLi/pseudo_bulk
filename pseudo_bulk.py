#import the necessary libraries
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad

def load_data(file_path):
    """
    Load and validate single-cell RNA-seq data from h5ad file
    
    @param file_path: Path to h5ad file
    @return: AnnData object with validated data structure
    """
    print("Loading data...")
    adata = sc.read_h5ad(file_path)
    
    # Validate required columns exist
    required_cols = ['donor_id', 'cell_type'] 
    missing = [col for col in required_cols if col not in adata.obs.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    print("Data loaded")    
    return adata

def get_summary(adata):
    """
    Get summary statistics about donors and cell types
    
    @param adata: AnnData object
    @return: Dictionary with summary statistics
    """
    donor_info = adata.obs['donor_id'].unique()
    cell_type_info = adata.obs['cell_type'].unique()
    
    return {
        'n_donors': len(donor_info),
        'n_cell_types': len(cell_type_info),
        'donors': donor_info,
        'cell_types': cell_type_info
    }

def get_celltype_specific_gene_matrix(adata, donor_id, cell_type):
    """
    Extract gene expression matrix for a specific donor and cell type.
    
    @param adata: AnnData single-cell RNA-seq data
    @param donor_id: Donor ID to extract
    @param cell_type: Cell type to extract
    @return: Gene expression matrix for the specified donor and cell type
    """
    # Create boolean filter for donor and cell type
    donor_celltype_filter = ((adata.obs['donor_id'] == donor_id) & (adata.obs['cell_type'] == cell_type)).values
    # Convert matrix to numpy array if it's a matrix type
    result = adata.X[donor_celltype_filter].copy() if not isinstance(adata.X, np.ndarray) else adata.X[donor_celltype_filter]
    if isinstance(result, np.matrix):
        result = np.array(result)
    return result

def calculate_cell_fractions(grouped):
    """
    Calculate cell counts and fractions for each donor-cell type combination
    
    @param grouped: Pandas groupby object by donor_id and cell_type
    @return: DataFrame with cell counts and fractions
    """
    cell_counts = grouped.size().reset_index(name='n_cells')
    donor_totals = cell_counts.groupby('donor_id', observed=True)['n_cells'].sum().reset_index(name='total_cells')
    cell_counts = cell_counts.merge(donor_totals, on='donor_id')
    cell_counts['fraction'] = cell_counts['n_cells'] / cell_counts['total_cells']
    return cell_counts

def create_pseudobulk(adata):
    """
    Create pseudobulk data aggregated by donor_id and cell_type.
    CPM normalized expression values are used.
    
    @param adata: AnnData single-cell RNA-seq data with donor_id and cell_type in obs
    @return: AnnData Pseudobulk data with donor_id and cell_type_info and cell type fractions
    """
    print("Creating pseudobulk data...")
    
    # Create groupby object for aggregation
    grouped = adata.obs.groupby(['donor_id', 'cell_type'], observed=True)
    cell_counts = calculate_cell_fractions(grouped)
    
    # Aggregate expression data
    pseudobulk_dict = {}
    celltype_matrices = {}
    
    for (donor, cell_type), group_indices in grouped.indices.items():
        # Sum expression values
        expr_sum = adata.X[group_indices].sum(axis=0).A1 if not isinstance(adata.X, np.ndarray) else adata.X[group_indices].sum(axis=0)
        
        # Convert to CPM
        total_counts = expr_sum.sum()
        if total_counts > 0:
            expr_cpm = (expr_sum * 1e6) / total_counts
        else:
            expr_cpm = expr_sum
            
        pseudobulk_dict[(donor, cell_type)] = expr_cpm
        
        # Store cell type matrix
        key = f"{donor}|||{cell_type}"
        matrix = get_celltype_specific_gene_matrix(adata, donor, cell_type)
        celltype_matrices[key] = matrix
    
    # Create observation dataframe
    obs_df = pd.DataFrame([key for key in pseudobulk_dict.keys()], columns=['donor_id', 'cell_type'])
    obs_df = obs_df.merge(cell_counts[['donor_id', 'cell_type', 'n_cells', 'fraction']], on=['donor_id', 'cell_type'])
    
    # Create pseudobulk AnnData
    pseudobulk_adata = ad.AnnData(
        X=np.vstack(list(pseudobulk_dict.values())),
        obs=obs_df,
        var=adata.var.copy(),
        uns={
            'celltype_matrices': celltype_matrices,
            'donor_celltype_pairs': list(grouped.indices.keys())
        }
    )
    
    print("Pseudobulk data done")
    return pseudobulk_adata

def save_pseudobulk_data(pbulk_adata, base_filename):
    """
    Save pseudobulk data and metadata to files
    
    @param pbulk_adata: AnnData pseudobulk object
    @param base_filename: Base name for output files
    """
    print(f"Saving {base_filename} data...")
    pbulk_adata.write_h5ad(f'data/{base_filename}_adata.h5ad')
    pbulk_adata.obs.to_csv(f'data/{base_filename}_metadata.csv', index=True)
    pbulk_adata.var.to_csv(f'data/{base_filename}_features.csv', index=True)
    print(f"{base_filename} data saved")

def save_expression_matrix(adata, output_filename):
    """
    Save the expression matrix from AnnData to a CSV file
    
    @param adata: AnnData object containing expression data
    @param output_filename: Name of the output CSV file
    """
    print(f"Saving expression matrix to {output_filename}...")
    
    # Get the index values - either from donor_id column or index
    if 'donor_id' in adata.obs.columns:
        index_values = adata.obs['donor_id'].values
    else:
        index_values = adata.obs.index.values
    
    # Convert X to a DataFrame
    expression_df = pd.DataFrame(
        adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
        index=index_values,  # Use either donor_id column or index
        columns=adata.var_names  # gene names
    )
    
    # Save to CSV
    expression_df.to_csv(output_filename)
    print(f"Expression matrix saved to {output_filename}")

def get_specific_cell_matrix(bulk_adata, donor, cell_type):
    """
    Helper function to retrieve the matrix for a specific donor and cell type
    
    @param pbulk_adata: AnnData pseudobulk data
    @param donor: Donor ID
    @param cell_type: Cell type
    @return: Gene expression matrix for the specified donor and cell type
    """
    key = f"{donor}|||{cell_type}"
    return bulk_adata.uns['celltype_matrices'].get(key)

def create_weighted_pseudobulk(adata):
    """
    Create weighted pseudobulk data per donor using weighted TPM normalization.
    For each cell type:
    1. Calculate raw counts
    2. Apply TPM normalization to raw counts
    3. Weight by cell type fraction
    4. Sum weighted TPM values across cell types
    
    @param adata: AnnData object containing single-cell data
    @return: AnnData object containing weighted pseudobulk data per donor
    """
    print("Creating weighted_pseudobulk data...")
    
    donors = adata.obs['donor_id'].unique()
    donor_profiles = []
    donor_metadata = []
    celltype_matrices = {}
    donor_celltype_pairs = []
    
    for donor in donors:
        donor_data = adata[adata.obs['donor_id'] == donor]
        donor_total_cells = donor_data.n_obs
        weighted_profile = None
        
        # For each cell type in this donor
        for ct in donor_data.obs['cell_type'].unique():
            ct_cells = donor_data[donor_data.obs['cell_type'] == ct]
            ct_fraction = ct_cells.n_obs / donor_total_cells
            
            # Get raw counts for this cell type
            ct_expr = ct_cells.X.sum(axis=0)
            if isinstance(ct_expr, np.matrix):
                ct_expr = np.array(ct_expr).flatten()
            
            # Calculate TPM for this cell type
            # First normalize by total counts in this cell type
            ct_total = ct_expr.sum()
            if ct_total > 0:
                ct_tpm = (ct_expr * 1e6) / ct_total
            else:
                ct_tpm = ct_expr
            
            # Store the raw expression matrix for this cell type
            key = f"{donor}|||{ct}"
            matrix = get_celltype_specific_gene_matrix(adata, donor, ct)
            celltype_matrices[key] = matrix
            donor_celltype_pairs.append((donor, ct))
            
            # Weight the TPM values by cell type fraction
            ct_weighted = ct_tpm * ct_fraction
            
            # Add to weighted profile
            weighted_profile = ct_weighted if weighted_profile is None else weighted_profile + ct_weighted
        
        # Store donor profile (no additional normalization needed)
        donor_profiles.append(weighted_profile.reshape(1, -1))
        
        # Store donor metadata
        donor_metadata.append({
            'donor_id': donor,
            'n_cells': donor_total_cells,
            'method': 'weighted_pseudobulk'
        })
    
    # Create observation dataframe with cell counts
    obs_df = pd.DataFrame(donor_metadata)
    
    weighted_pbulk = ad.AnnData(
        X=np.vstack(donor_profiles),
        var=adata.var.copy(),
        obs=obs_df,  # Don't set index here
        uns={
            'celltype_matrices': celltype_matrices,
            'donor_celltype_pairs': donor_celltype_pairs
        }
    )
    print("Weighted pseudobulk data done")
    return weighted_pbulk

if __name__ == '__main__':
    # Load data
    adata = load_data('data/08984b3c-3189-4732-be22-62f1fe8f15a4.h5ad')
    
    # Get summary statistics
    stats = get_summary(adata)
    print(f"Found {stats['n_donors']} donors and {stats['n_cell_types']} cell types")
    
    # Create and save pseudobulk data
    pseudobulk_adata = create_pseudobulk(adata)
    save_pseudobulk_data(pseudobulk_adata, 'pseudobulk')
    print("Pseudobulk shape:", pseudobulk_adata.X.shape)
    save_expression_matrix(pseudobulk_adata, 'data/pseudobulk_expression.csv')
    
    # Create weighted pseudobulk
    weighted_pbulk = create_weighted_pseudobulk(adata)
    save_pseudobulk_data(weighted_pbulk, 'weighted_pseudobulk')
    print("Weighted pseudobulk shape:", weighted_pbulk.X.shape)
    save_expression_matrix(weighted_pbulk, 'data/weighted_pseudobulk_expression.csv')

    # Get matrix for a specific donor and cell type
    print(get_specific_cell_matrix(pseudobulk_adata, '1_1', 'natural killer cell'))
    print(get_specific_cell_matrix(weighted_pbulk, '1_1', 'natural killer cell'))