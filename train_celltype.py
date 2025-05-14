import os
import logging
import jax
import jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import pandas as pd
from multiomics_open_research.bulk_rna_bert.pretrained import get_pretrained_model
from multiomics_open_research.bulk_rna_bert.preprocess import preprocess_rna_seq_for_bulkrnabert
from contrastive_loss import compute_contrastive_loss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Track the current round number
ROUND = 1 # Change this number for each new run

# Optimize JAX configuration for GPU
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_disable_jit', False)  # Enable JIT compilation
jax.config.update('jax_threefry_partitionable', True)  # Enable better parallelization

def load_data():
    """Load celltype data."""
    return pd.read_csv("data/celltype_specific_2d_matrix.csv", index_col=0)

def save_embeddings(embeddings, filename):
    """Save embeddings to a numpy file, accumulating them if the file exists."""
    try:
        # Convert current batch to numpy array
        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # If file exists, load and concatenate
        if os.path.exists(filename):
            existing_embeddings = np.load(filename)
            embeddings_np = np.concatenate([existing_embeddings, embeddings_np], axis=0)
        
        # Save the accumulated embeddings
        np.save(filename, embeddings_np, allow_pickle=False)
        
        # Verify the save
        loaded = np.load(filename)
        if loaded.shape != embeddings_np.shape:
            raise ValueError(f"Saved shape {loaded.shape} doesn't match expected shape {embeddings_np.shape}")
            
        logging.info(f"Saved embeddings to {filename} with shape {embeddings_np.shape}")
    except Exception as e:
        logging.error(f"Error saving embeddings to {filename}: {str(e)}")
        raise

def save_checkpoint(params, filename):
    """Save model parameters to a numpy file."""
    np.save(filename, params)
    logging.info(f"Saved checkpoint to {filename}")

def load_checkpoint(filename):
    """Load model parameters from a numpy file."""
    params = np.load(filename, allow_pickle=True).item()
    logging.info(f"Loaded checkpoint from {filename}")
    return params

def create_zero_grads(params):
    """Create zero gradients matching the parameter structure."""
    if isinstance(params, dict):
        return {k: create_zero_grads(v) for k, v in params.items()}
    else:
        return jnp.zeros_like(params)

def train_step(params, opt_state, batch, forward_fn, optimizer, rng_key, pseudobulk_donors, celltype_donors):
    """Single training step."""
    # Generate embeddings for the current batch
    outs = forward_fn.apply(params, rng_key, batch)
    batch_embeddings = outs["embeddings_4"].mean(axis=1)
    
    # Load current embeddings for contrastive loss computation
    if ROUND == 1:
        pseudobulk_embeddings = np.load(f'data/mean_embeddings.npy')
    else:
        pseudobulk_embeddings = np.load(f'embeddings/pseudobulk_embeddings_round_{ROUND-1}.npy')
    
    # Get the donor IDs for the current batch
    batch_indices = [i for i in range(len(celltype_donors)) if i < len(batch_embeddings)]
    batch_donors = [celltype_donors[i] for i in batch_indices]
    
    # Compute loss for the current batch against all pseudobulk embeddings
    _, celltype_loss = compute_contrastive_loss(
        pseudobulk_embeddings,
        batch_embeddings,
        pseudobulk_donors,
        batch_donors
    )
    
    # Create a gradient structure matching the model parameters
    grads = create_zero_grads(params)
    
    # Apply gradients to embedding layer
    if 'embedding' in grads:
        # Compute gradients for celltype embeddings only
        celltype_grads = jax.grad(lambda x: compute_contrastive_loss(pseudobulk_embeddings, x, pseudobulk_donors, batch_donors)[1])(batch_embeddings)
        grads['embedding'] = jnp.array(celltype_grads)
    
    # Update parameters using gradients
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, batch_embeddings, celltype_loss

def main():
    # Create directories
    os.makedirs('embeddings', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Get pretrained model
    logging.info("Loading pretrained model...")
    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name="bulk_rna_bert_tcga",
        embeddings_layers_to_save=(4,),
        checkpoint_directory="multiomics-open-research/checkpoints/",
    )
    forward_fn = hk.transform(forward_fn)
    
    # Try to load previous parameters if they exist
    current_checkpoint = f'checkpoints/celltype_round_{ROUND}.npy'
    previous_checkpoint = f'checkpoints/celltype_round_{ROUND-1}.npy'
    
    if os.path.exists(current_checkpoint):
        logging.info(f"Loading current round parameters from {current_checkpoint}")
        parameters = load_checkpoint(current_checkpoint)
    elif os.path.exists(previous_checkpoint):
        logging.info(f"Loading previous round parameters from {previous_checkpoint}")
        parameters = load_checkpoint(previous_checkpoint)
    else:
        logging.info("No previous parameters found, using initial parameters")
    
    # Initialize optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(1e-4)
    )
    opt_state = optimizer.init(parameters)
    
    # Load data
    logging.info("Loading data...")
    celltype_df = load_data()
    celltype_array = preprocess_rna_seq_for_bulkrnabert(celltype_df, config)
    celltype_tokens = jnp.asarray(tokenizer.batch_tokenize(celltype_array), dtype=jnp.int32)
    
    # Get donor IDs
    celltype_donors = celltype_df.index.tolist()
    pseudobulk_df = pd.read_csv("data/processed_pseudobulk_expression_W.csv", index_col=0)
    pseudobulk_donors = pseudobulk_df.index.tolist()
    
    # Training parameters
    batch_size = 1
    total_loss = 0
    all_embeddings = []
    processed_samples = set()  # Track processed samples
    
    # Process in batches
    for i in range(0, len(celltype_tokens), batch_size):
        batch = celltype_tokens[i:i + batch_size]
        current_donor = celltype_donors[i]
        
        # Skip if we've already processed this sample
        if current_donor in processed_samples:
            continue
            
        processed_samples.add(current_donor)
        
        # Training step
        rng_key = jax.random.PRNGKey(i)
        parameters, opt_state, batch_embeddings, celltype_loss = train_step(
            parameters, opt_state, batch, forward_fn, optimizer, rng_key,
            pseudobulk_donors, celltype_donors
        )
        
        total_loss += float(celltype_loss)
        all_embeddings.append(batch_embeddings)
        
        if (i // batch_size) % 10 == 0:
            avg_loss = total_loss / len(processed_samples)
            logging.info(f"Processed {len(processed_samples)}/{len(celltype_donors)} samples, Average loss: {avg_loss:.4f}")
    
    # Concatenate all embeddings and save
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(f'embeddings/celltype_embeddings_round_{ROUND}.npy', all_embeddings)
    
    # Save final checkpoint
    save_checkpoint(parameters, f'checkpoints/celltype_round_{ROUND}.npy')
    logging.info(f"Training completed for round {ROUND}!")

if __name__ == "__main__":
    main() 