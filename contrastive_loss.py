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

def sparse_categorical_crossentropy(labels, logits):
    """
    Compute sparse categorical cross-entropy loss manually.
    
    Args:
        labels: Integer labels [batch_size]
        logits: Logits from model [batch_size, num_classes]
        
    Returns:
        Loss value for each sample
    """
    # Convert labels to one-hot encoding
    num_classes = logits.shape[-1]
    one_hot_labels = jax.nn.one_hot(labels, num_classes)
    
    # Compute softmax
    log_probs = jax.nn.log_softmax(logits)
    
    # Compute cross-entropy
    return -jnp.sum(one_hot_labels * log_probs, axis=-1)

def compute_contrastive_loss(pseudobulk_embeddings, celltype_embeddings, temperature=0.07):
    """
    Compute contrastive loss between pseudobulk and celltype embeddings.
    
    Args:
        pseudobulk_embeddings: Embeddings from pseudobulk data [batch_size, embedding_dim]
        celltype_embeddings: Embeddings from celltype-specific data [batch_size, embedding_dim]
        temperature: Temperature parameter for softmax scaling
        
    Returns:
        Tuple of (pseudobulk_loss, celltype_loss) where each is the mean loss for that direction
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
    loss_pseudobulk = sparse_categorical_crossentropy(labels, similarity_matrix)
    loss_celltype = sparse_categorical_crossentropy(labels, similarity_matrix.T)
    
    # Return both losses separately
    return loss_pseudobulk.mean(), loss_celltype.mean()

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
    
    # Load embeddings
    logging.info("Loading embeddings...")
    pseudobulk_embeddings = np.load('data/mean_embeddings.npy')
    celltype_embeddings = np.load('data/celltype_specific_embeddings.npy')
    
    # Convert to JAX arrays
    pseudobulk_embeddings = jnp.array(pseudobulk_embeddings)
    celltype_embeddings = jnp.array(celltype_embeddings)
    
    # Compute loss
    logging.info("Computing contrastive loss...")
    pseudobulk_loss, celltype_loss = compute_contrastive_loss(pseudobulk_embeddings, celltype_embeddings)
    
    # Save losses
    loss_file = 'losses/current_losses.txt'
    with open(loss_file, 'w') as f:
        f.write(f"Pseudobulk loss: {float(pseudobulk_loss):.4f}\n")
        f.write(f"Celltype loss: {float(celltype_loss):.4f}\n")
    logging.info(f"Saved losses to {loss_file}")
    logging.info(f"Pseudobulk loss: {float(pseudobulk_loss):.4f}")
    logging.info(f"Celltype loss: {float(celltype_loss):.4f}")

if __name__ == "__main__":
    main()
