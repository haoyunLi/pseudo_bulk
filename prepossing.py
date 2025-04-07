import pandas as pd
from typing import Tuple, List

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
    # Get gene columns from processed data (excluding the first column which is sample IDs)
    processed_genes = processed_data.columns[1:].tolist()
    
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

def process_pseudobulk_data(pseudobulk_data_path: str, common_genes_path: str) -> pd.DataFrame:
    """
    Process pseudobulk data to match common genes list exactly, adding missing genes with zero expression.
    
    @param pseudobulk_data_path: Path to the pseudobulk data CSV file
    @param common_genes_path: Path to the common gene IDs file
    @return: Processed DataFrame with all common genes in correct order
    """
    # Read the pseudobulk data
    pseudobulk_data = pd.read_csv(pseudobulk_data_path)
    
    # Get the common gene ids
    common_genes = load_common_genes(common_genes_path)
    
    # Get gene ids from pseudobulk data (excluding the first column which is likely sample IDs)
    pseudobulk_genes = pseudobulk_data.columns[1:].tolist()
    
    # Check for genes in pseudobulk but not in common genes
    missing_in_common = set(pseudobulk_genes) - set(common_genes)
    if missing_in_common:
        print(f"Warning: {len(missing_in_common)} genes in pseudobulk data are not in common genes list")
    
    # Check for genes in common list but not in pseudobulk
    missing_in_pseudobulk = set(common_genes) - set(pseudobulk_genes)
    if missing_in_pseudobulk:
        print(f"Note: {len(missing_in_pseudobulk)} genes in common list are not in pseudobulk data")
        print("These genes will be added with zero expression values")
    
    # Keep the sample ID column
    sample_id_col = pseudobulk_data.columns[0]
    processed_data = pseudobulk_data[[sample_id_col]].copy()
    
    # Create a dictionary to store gene columns
    gene_columns = {}
    
    # Add genes in the order of common_genes
    for gene in common_genes:
        if gene in pseudobulk_genes:
            # If gene exists in pseudobulk, use its expression values
            gene_columns[gene] = pseudobulk_data[gene]
        else:
            # If gene is missing, add it with zero expression
            gene_columns[gene] = pd.Series(0, index=pseudobulk_data.index)
    
    # Concatenate all gene columns at once
    gene_df = pd.DataFrame(gene_columns)
    processed_data = pd.concat([processed_data, gene_df], axis=1)
    
    return processed_data

def save_processed_data(processed_data: pd.DataFrame, output_path: str) -> None:
    """
    Save the processed data to a CSV file.
    
    @param processed_data: Processed DataFrame to save
    @param output_path: Path where to save the processed data
    """
    processed_data.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    # Load the data
    pseudobulk_data_path = 'data/weighted_pseudobulk_expression.csv'
    common_genes_path = 'data/common_gene_id.txt'
    output_path = 'data/processed_pseudobulk_expression.csv'

    # Process the data
    processed_data = process_pseudobulk_data(pseudobulk_data_path, common_genes_path)
    
    # Load common genes for verification
    common_genes = load_common_genes(common_genes_path)
    
    # Verify gene IDs and order
    gene_ids_match, gene_order_match = verify_gene_ids_and_order(processed_data, common_genes)
    
    # If order doesn't match, reorder the columns
    if not gene_order_match:
        processed_data = processed_data[processed_data.columns[0:1].tolist() + common_genes]
    
    # Save the processed data
    save_processed_data(processed_data, output_path)



    
    



