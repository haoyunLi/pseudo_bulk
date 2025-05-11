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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Track the current round number
ROUND = 2  # Change this number for each new run

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
    """Save embeddings to a numpy file."""
    embeddings_np = np.array(embeddings)
    np.save(filename, embeddings_np)
    logging.info(f"Saved embeddings to {filename}")

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

def train_step(params, opt_state, batch, forward_fn, optimizer, rng_key):
    """Single training step."""
    # Generate embeddings
    outs = forward_fn.apply(params, rng_key, batch)
    embeddings = outs["embeddings_4"].mean(axis=1)
    
    # Load current loss and gradients from files
    with open(f'losses/current_losses_round_{ROUND}.txt', 'r') as f:
        for line in f:
            if line.startswith('Celltype loss:'):
                loss = float(line.split(':')[1].strip())
                break
    
    # Load gradients
    raw_grads = np.load(f'losses/celltype_grads_round_{ROUND}.npy')
    
    # Create a gradient structure matching the model parameters
    grads = create_zero_grads(params)
    
    # Apply raw gradients to embedding layer
    if 'embedding' in grads:
        grads['embedding'] = jnp.array(raw_grads)
    
    # Update parameters using gradients
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, embeddings

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
    #previous_checkpoint = f'checkpoints/celltype_final.npy'
        
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
    
    # Training parameters
    batch_size = 1
    
    # Process in batches
    for i in range(0, len(celltype_tokens), batch_size):
        batch = celltype_tokens[i:i + batch_size]
        
        # Training step
        rng_key = jax.random.PRNGKey(i)
        parameters, opt_state, embeddings = train_step(
            parameters, opt_state, batch, forward_fn, optimizer, rng_key
        )
        
        # Save embeddings
        save_embeddings(embeddings, f'embeddings/celltype_embeddings_round_{ROUND}.npy')
        
        if (i // batch_size) % 10 == 0:
            logging.info(f"Processed batch {i//batch_size + 1}/{(len(celltype_tokens) + batch_size - 1)//batch_size}")
    
    # Save final checkpoint
    save_checkpoint(parameters, f'checkpoints/celltype_round_{ROUND}.npy')
    logging.info(f"Training completed for round {ROUND}!")

if __name__ == "__main__":
    main() 