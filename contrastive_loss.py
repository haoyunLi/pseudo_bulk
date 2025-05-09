import jax
import jax.numpy as jnp
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def compute_contrastive_loss(pseudobulk_embeddings, celltype_embeddings, temperature=0.07):
    """
    Compute contrastive loss between pseudobulk and celltype embeddings.
    
    Args:
        pseudobulk_embeddings: Embeddings from pseudobulk data [batch_size, embedding_dim]
        celltype_embeddings: Embeddings from celltype-specific data [batch_size, embedding_dim]
        temperature: Temperature parameter for softmax scaling
        
    Returns:
        Contrastive loss value
    """
    # Normalize embeddings
    pseudobulk_norm = jnp.linalg.norm(pseudobulk_embeddings, axis=1, keepdims=True)
    celltype_norm = jnp.linalg.norm(celltype_embeddings, axis=1, keepdims=True)
    
    pseudobulk_normalized = pseudobulk_embeddings / pseudobulk_norm
    celltype_normalized = celltype_embeddings / celltype_norm
    
    # Compute similarity matrix
    similarity_matrix = jnp.matmul(pseudobulk_normalized, celltype_normalized.T) / temperature
    
    # Create labels (diagonal elements are positive pairs)
    labels = jnp.arange(similarity_matrix.shape[0])
    
    # Compute loss for both directions
    loss_pseudobulk = jax.nn.sparse_categorical_crossentropy_with_logits(
        labels, similarity_matrix
    )
    loss_celltype = jax.nn.sparse_categorical_crossentropy_with_logits(
        labels, similarity_matrix.T
    )
    
    # Average the losses
    total_loss = (loss_pseudobulk.mean() + loss_celltype.mean()) / 2
    
    return total_loss

def compute_gradients(loss_fn, params, pseudobulk_embeddings, celltype_embeddings):
    """
    Compute gradients of the contrastive loss with respect to the embeddings.
    
    Args:
        loss_fn: Function that computes the contrastive loss
        params: Model parameters
        pseudobulk_embeddings: Embeddings from pseudobulk data
        celltype_embeddings: Embeddings from celltype-specific data
        
    Returns:
        Gradients for both pseudobulk and celltype embeddings
    """
    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))
    (loss, (pseudobulk_grads, celltype_grads)) = grad_fn(
        pseudobulk_embeddings, celltype_embeddings
    )
    return loss, pseudobulk_grads, celltype_grads

def main():
    # Create directory for losses
    os.makedirs('losses', exist_ok=True)
    
    # Load embeddings
    logging.info("Loading embeddings...")
    pseudobulk_embeddings = np.load('data/mean_embeddings.npy')
    celltype_embeddings = np.load('data/celltype_specific_embeddings.npy')
    
    # Convert to JAX arrays
    pseudobulk_embeddings = jnp.array(pseudobulk_embeddings)
    celltype_embeddings = jnp.array(celltype_embeddings)
    
    # Compute loss
    logging.info("Computing contrastive loss...")
    loss = compute_contrastive_loss(pseudobulk_embeddings, celltype_embeddings)
    
    # Save loss
    loss_file = 'losses/current_loss.txt'
    with open(loss_file, 'w') as f:
        f.write(str(float(loss)))
    logging.info(f"Saved loss to {loss_file}")
    logging.info(f"Current loss: {float(loss):.4f}")

if __name__ == "__main__":
    main()
