import jax
import jax.numpy as jnp
import numpy as np
import logging
import os
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Track the current round number
ROUND = 1  # Change this number for each new run

def sparse_categorical_crossentropy(labels, logits):
    """
    Compute sparse categorical cross-entropy loss manually.
    
    Args:
        labels: Integer labels [batch_size]
        logits: Logits from model [batch_size, num_classes]
        
    Returns:
        Loss value for each sample
    """
    # Ensure inputs are the right shape
    if labels.ndim != 1:
        raise ValueError(f"Labels must be 1D array, got shape {labels.shape}")
    if logits.ndim != 2:
        raise ValueError(f"Logits must be 2D array, got shape {logits.shape}")
    if labels.shape[0] != logits.shape[0]:
        raise ValueError(f"Batch size mismatch: labels {labels.shape[0]} vs logits {logits.shape[0]}")
    
    # Convert labels to one-hot encoding
    num_classes = logits.shape[-1]
    one_hot_labels = jax.nn.one_hot(labels, num_classes)
    
    # Compute softmax and cross-entropy in a numerically stable way
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.sum(one_hot_labels * log_probs, axis=-1)

def extract_donor_id(celltype_id):
    """Extract donor ID from celltype ID (format: 'celltype|donor_id')"""
    return celltype_id.split('|')[1]

def compute_contrastive_loss(pseudobulk_embeddings, celltype_embeddings, pseudobulk_donors, celltype_donors, temperature=0.07):
    """
    Compute contrastive loss between pseudobulk and celltype embeddings.
    
    Args:
        pseudobulk_embeddings: Embeddings from pseudobulk data [n_samples, embedding_dim]
        celltype_embeddings: Embeddings from celltype-specific data [n_celltypes, embedding_dim]
        pseudobulk_donors: Donor IDs for pseudobulk samples
        celltype_donors: Donor IDs for celltype samples
        temperature: Temperature parameter for softmax scaling
        
    Returns:
        Tuple of (pseudobulk_loss, celltype_loss) where each is the mean loss for that direction
    """
    # Validate input shapes
    if pseudobulk_embeddings.shape[1] != celltype_embeddings.shape[1]:
        raise ValueError(f"Embedding dimension mismatch: pseudobulk {pseudobulk_embeddings.shape[1]} vs celltype {celltype_embeddings.shape[1]}")
    
    # Create mapping from donor ID to sample index
    donor_to_index = {donor: idx for idx, donor in enumerate(pseudobulk_donors)}
    
    # Create sample indices from donor IDs
    sample_indices = jnp.array([donor_to_index[extract_donor_id(donor)] for donor in celltype_donors])
    
    # Log the mapping (only during non-gradient computation)
    if not jax.core.is_abstract(pseudobulk_embeddings):
        unique_samples = jnp.unique(sample_indices)
        logging.info(f"Number of unique samples: {len(unique_samples)}")
        for sample_idx in unique_samples:
            n_celltypes = jnp.sum(sample_indices == sample_idx)
            donor = pseudobulk_donors[sample_idx]
            logging.info(f"Sample {donor}: {n_celltypes} celltype embeddings")
    
    # Normalize embeddings
    pseudobulk_norm = jnp.linalg.norm(pseudobulk_embeddings, axis=1, keepdims=True)
    celltype_norm = jnp.linalg.norm(celltype_embeddings, axis=1, keepdims=True)
    
    # Add small epsilon to avoid division by zero
    eps = 1e-8
    pseudobulk_normalized = pseudobulk_embeddings / (pseudobulk_norm + eps)
    celltype_normalized = celltype_embeddings / (celltype_norm + eps)
    
    # Compute similarity matrix [n_pseudobulk, n_celltypes]
    similarity_matrix = jnp.matmul(pseudobulk_normalized, celltype_normalized.T) / temperature
    
    # Log shapes (only during non-gradient computation)
    if not jax.core.is_abstract(pseudobulk_embeddings):
        logging.info(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    # Create labels for each direction
    pseudobulk_labels = jnp.zeros(similarity_matrix.shape, dtype=jnp.float32)
    for i, donor in enumerate(pseudobulk_donors):
        matching_celltypes = jnp.array([extract_donor_id(d) == donor for d in celltype_donors], dtype=jnp.float32)
        pseudobulk_labels = pseudobulk_labels.at[i].set(matching_celltypes)
    
    celltype_labels = sample_indices
    
    # Log shapes (only during non-gradient computation)
    if not jax.core.is_abstract(pseudobulk_embeddings):
        logging.info(f"Pseudobulk labels shape: {pseudobulk_labels.shape}")
        logging.info(f"Celltype labels shape: {celltype_labels.shape}")
    
    # Compute loss for both directions
    positive_pairs = jnp.sum(pseudobulk_labels, axis=1, keepdims=True)
    pseudobulk_loss = -jnp.sum(pseudobulk_labels * jax.nn.log_softmax(similarity_matrix, axis=1)) / jnp.sum(positive_pairs)
    
    # Compute celltype loss using the same approach as pseudobulk loss
    celltype_one_hot = jax.nn.one_hot(celltype_labels, similarity_matrix.shape[0])
    celltype_positive_pairs = jnp.sum(celltype_one_hot, axis=1, keepdims=True)
    celltype_loss = -jnp.sum(celltype_one_hot * jax.nn.log_softmax(similarity_matrix.T, axis=1)) / jnp.sum(celltype_positive_pairs)
    
    # Log losses (only during non-gradient computation)
    if not jax.core.is_abstract(pseudobulk_embeddings):
        logging.info(f"pseudobulk loss: {float(pseudobulk_loss):.4f}")
        logging.info(f"celltype loss: {float(celltype_loss):.4f}")
    
    return pseudobulk_loss, celltype_loss

def compute_gradients(loss_fn, params, pseudobulk_embeddings, celltype_embeddings):
    """
    Compute gradients of the contrastive loss with respect to the embeddings.
    
    Args:
        loss_fn: Function that computes the contrastive loss
        params: Model parameters
        pseudobulk_embeddings: Embeddings from pseudobulk data
        celltype_embeddings: Embeddings from celltype-specific data
        
    Returns:
        Tuple of (pseudobulk_loss, celltype_loss, pseudobulk_grads, celltype_grads)
    """
    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))
    ((pseudobulk_loss, celltype_loss), (pseudobulk_grads, celltype_grads)) = grad_fn(
        pseudobulk_embeddings, celltype_embeddings
    )
    return pseudobulk_loss, celltype_loss, pseudobulk_grads, celltype_grads

def main():
    # Create directory for losses
    os.makedirs('losses', exist_ok=True)
    
    # Load embeddings and donor IDs
    logging.info("Loading embeddings and donor IDs...")
    
    # Load embeddings from NPY files
    pseudobulk_embeddings = np.load(f'embeddings/pseudobulk_embeddings_round_{ROUND-1}.npy')
    celltype_embeddings = np.load(f'embeddings/celltype_embeddings_round_{ROUND-1}.npy')
    
    #pseudobulk_embeddings = np.load(f'data/mean_embeddings.npy')
    #celltype_embeddings = np.load(f'data/celltype_specific_embeddings.npy')
    
    # Load donor IDs from CSV files
    pseudobulk_df = pd.read_csv('data/mean_embeddings.csv', index_col=0)
    celltype_df = pd.read_csv('data/celltype_specific_embeddings.csv', index_col=0)
    
    # Get donor IDs
    pseudobulk_donors = pseudobulk_df.index.tolist()
    celltype_donors = celltype_df.index.tolist()
    
    # Log shapes for debugging
    logging.info(f"Pseudobulk embeddings shape: {pseudobulk_embeddings.shape}")
    logging.info(f"Celltype embeddings shape: {celltype_embeddings.shape}")
    logging.info(f"Number of pseudobulk samples: {len(pseudobulk_donors)}")
    logging.info(f"Number of celltype samples: {len(celltype_donors)}")
    
    # Convert to JAX arrays
    pseudobulk_embeddings = jnp.array(pseudobulk_embeddings)
    celltype_embeddings = jnp.array(celltype_embeddings)
    
    # Compute loss and gradients
    logging.info("Computing contrastive loss and gradients...")
    pseudobulk_loss, celltype_loss = compute_contrastive_loss(
        pseudobulk_embeddings, 
        celltype_embeddings,
        pseudobulk_donors,
        celltype_donors
    )
    
    # Log losses
    logging.info(f"Pseudobulk loss: {float(pseudobulk_loss):.4f}")
    logging.info(f"Celltype loss: {float(celltype_loss):.4f}")

if __name__ == "__main__":
    main()
