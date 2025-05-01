import jax
import jax.numpy as jnp
import numpy as np

def contrastive_loss(pseudobulk_embeddings, celltype_embeddings, temperature=0.07):
    """
    Compute contrastive loss between pseudobulk and celltype-specific embeddings using JAX
    
    Args:
        pseudobulk_embeddings (jnp.ndarray): Embeddings from pseudobulk data
        celltype_embeddings (jnp.ndarray): Embeddings from celltype-specific data
        temperature (float): Temperature parameter for scaling the similarity scores
        
    Returns:
        float: Contrastive loss value
    """
    # Normalize embeddings
    pseudobulk_embeddings = pseudobulk_embeddings / jnp.linalg.norm(pseudobulk_embeddings, axis=1, keepdims=True)
    celltype_embeddings = celltype_embeddings / jnp.linalg.norm(celltype_embeddings, axis=1, keepdims=True)
    
    # Compute similarity matrix
    similarity_matrix = jnp.matmul(pseudobulk_embeddings, celltype_embeddings.T) / temperature
    
    # Create labels for positive pairs (diagonal elements)
    labels = jnp.arange(similarity_matrix.shape[0])
    
    # Compute logits for cross entropy
    logits_pseudobulk = similarity_matrix
    logits_celltype = similarity_matrix.T
    
    # Compute cross entropy loss for both directions
    def cross_entropy(logits, labels):
        log_probs = jax.nn.log_softmax(logits, axis=1)
        return -jnp.mean(jnp.take_along_axis(log_probs, labels[:, None], axis=1))
    
    loss_pseudobulk = cross_entropy(logits_pseudobulk, labels)
    loss_celltype = cross_entropy(logits_celltype, labels)
    
    # Total loss is the average of both directions
    total_loss = (loss_pseudobulk + loss_celltype) / 2
    
    return total_loss

def compute_contrastive_loss(pseudobulk_data, celltype_data, model_fn, params, rng_key):
    """
    Helper function to compute contrastive loss for a batch of data
    
    Args:
        pseudobulk_data (jnp.ndarray): Pseudobulk data batch
        celltype_data (jnp.ndarray): Celltype-specific data batch
        model_fn: Haiku transformed model function
        params: Model parameters
        rng_key: JAX random key
        
    Returns:
        float: Contrastive loss value
    """
    # Generate embeddings
    pseudobulk_outs = model_fn.apply(params, rng_key, pseudobulk_data)
    celltype_outs = model_fn.apply(params, rng_key, celltype_data)
    
    # Get mean embeddings from layer 4
    pseudobulk_embeddings = pseudobulk_outs["embeddings_4"].mean(axis=1)
    celltype_embeddings = celltype_outs["embeddings_4"].mean(axis=1)
    
    # Compute loss
    loss = contrastive_loss(pseudobulk_embeddings, celltype_embeddings)
    
    return loss
