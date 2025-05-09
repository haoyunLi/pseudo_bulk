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

def load_data():
    """Load pseudobulk data."""
    return pd.read_csv("data/processed_pseudobulk_expression_W.csv", index_col=0)

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

def train_step(params, opt_state, batch, forward_fn, optimizer, rng_key):
    """Single training step."""
    # Generate embeddings
    outs = forward_fn.apply(params, rng_key, batch)
    embeddings = outs["embeddings_4"].mean(axis=1)
    
    # Load current loss
    with open('losses/current_loss.txt', 'r') as f:
        loss = float(f.read().strip())
    
    # Update parameters using the loss
    updates, opt_state = optimizer.update(loss, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, embeddings

def main():
    # Create directories
    os.makedirs('embeddings', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Get pretrained model
    logging.info("Loading pretrained model...")
    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name="bulk_rna_bert_gtex_encode",
        embeddings_layers_to_save=(4,),
        checkpoint_directory="multiomics-open-research/checkpoints/",
        compute_dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        output_dtype=jnp.float32
    )
    forward_fn = hk.transform(forward_fn)
    
    # Initialize optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(1e-4)
    )
    opt_state = optimizer.init(parameters)
    
    # Load data
    logging.info("Loading data...")
    pseudobulk_df = load_data()
    pseudobulk_array = preprocess_rna_seq_for_bulkrnabert(pseudobulk_df, config)
    pseudobulk_tokens = jnp.asarray(tokenizer.batch_tokenize(pseudobulk_array), dtype=jnp.int32)
    
    # Training parameters
    batch_size = 1
    
    # Process in batches
    for i in range(0, len(pseudobulk_tokens), batch_size):
        batch = pseudobulk_tokens[i:i + batch_size]
        
        # Training step
        rng_key = jax.random.PRNGKey(i)
        parameters, opt_state, embeddings = train_step(
            parameters, opt_state, batch, forward_fn, optimizer, rng_key
        )
        
        # Save embeddings
        save_embeddings(embeddings, 'embeddings/pseudobulk_embeddings.npy')
        
        if (i // batch_size) % 10 == 0:
            logging.info(f"Processed batch {i//batch_size + 1}/{(len(pseudobulk_tokens) + batch_size - 1)//batch_size}")
    
    # Save final checkpoint
    save_checkpoint(parameters, 'checkpoints/pseudobulk_final.npy')
    logging.info("Training completed!")

if __name__ == "__main__":
    main() 