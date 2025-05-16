import haiku as hk
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from multiomics_open_research.bulk_rna_bert.pretrained import get_pretrained_model
from multiomics_open_research.bulk_rna_bert.preprocess import preprocess_rna_seq_for_bulkrnabert
import optax
import logging
import os
import pickle
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh
from jax.experimental.pjit import pjit
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Configure JAX for distributed training
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'  # Force JAX to see 4 GPUs
jax.config.update('jax_platform_name', 'cuda')
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_threefry_partitionable', True)
jax.config.update('jax_enable_custom_prng', True)

# Initialize distributed training
devices = jax.devices()
print(f"Available devices: {devices}")
NUM_DEVICES = len(devices)

# Create mesh for model parallelism
mesh = Mesh(np.array(devices).reshape(-1), ('model',))
mesh_shape = (NUM_DEVICES,)

# Define partition specs
param_spec = P('model')  # Parameters are sharded across model dimension
batch_spec = P(None)     # Batch dimension is not sharded

def compute_contrastive_loss(embeddings1, embeddings2, labels1, labels2, temperature=0.07):
    """
    Compute contrastive loss between two sets of embeddings.
    
    Args:
        embeddings1: First set of embeddings (pseudobulk)
        embeddings2: Second set of embeddings (celltype)
        labels1: Labels for first set (donor IDs)
        labels2: Labels for second set (donor IDs)
        temperature: Temperature parameter for softmax
    
    Returns:
        loss: Contrastive loss value
    """
    # Normalize embeddings
    embeddings1 = embeddings1 / jnp.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / jnp.linalg.norm(embeddings2, axis=1, keepdims=True)
    
    # Compute similarity matrix
    similarity_matrix = jnp.matmul(embeddings1, embeddings2.T) / temperature
    
    # Create positive mask (same donor)
    positive_mask = jnp.equal(labels1[:, None], labels2[None, :])
    
    # Compute loss
    exp_sim = jnp.exp(similarity_matrix)
    positive_sim = jnp.sum(exp_sim * positive_mask, axis=1)
    negative_sim = jnp.sum(exp_sim * (1 - positive_mask), axis=1)
    
    loss = -jnp.mean(jnp.log(positive_sim / (positive_sim + negative_sim + 1e-8)))
    
    return loss

def shard_params(params):
    """Shard parameters across devices."""
    def shard_param(param):
        if len(param.shape) > 1:  # Only shard parameters with multiple dimensions
            # For transformer parameters, shard along the hidden dimension
            if 'embeddings' in str(param):  # Special handling for embeddings
                return param
            else:  # For all other parameters
                # Try to shard along the largest dimension
                dim_sizes = param.shape
                max_dim = max(range(len(dim_sizes)), key=lambda i: dim_sizes[i])
                if dim_sizes[max_dim] >= NUM_DEVICES:
                    # Calculate shard size and padding if needed
                    shard_size = (dim_sizes[max_dim] + NUM_DEVICES - 1) // NUM_DEVICES
                    if dim_sizes[max_dim] % NUM_DEVICES != 0:
                        pad_size = shard_size * NUM_DEVICES - dim_sizes[max_dim]
                        padding = [(0, 0)] * max_dim + [(0, pad_size)] + [(0, 0)] * (len(param.shape) - max_dim - 1)
                        param = jnp.pad(param, padding)
                    
                    # Split and distribute across devices
                    splits = jnp.split(param, NUM_DEVICES, axis=max_dim)
                    return jax.device_put_sharded(splits, devices)
        return param
    
    return jax.tree_map(shard_param, params)

def train_step(params, opt_state, pseudobulk_batch, celltype_batch, forward_fn, optimizer, rng_key, pseudobulk_donors, celltype_donors):
    """Single training step computing contrastive loss for both modalities."""
    try:
        # Shard parameters across devices
        sharded_params = shard_params(params)
        
        def loss_fn(params):
            # Forward pass for both batches
            pseudobulk_outs = forward_fn.apply(params, rng_key, pseudobulk_batch)
            celltype_outs = forward_fn.apply(params, rng_key, celltype_batch)
            
            # Get embeddings and handle dimensions properly
            pseudobulk_embeddings = pseudobulk_outs["embeddings_4"].mean(axis=(0, 1))  # Average across attention heads and sequence length
            celltype_embeddings = celltype_outs["embeddings_4"].mean(axis=(0, 1))      # Average across attention heads and sequence length
            
            # Compute contrastive loss
            loss = compute_contrastive_loss(
                pseudobulk_embeddings,
                celltype_embeddings,
                pseudobulk_donors,
                celltype_donors
            )
            
            return loss, (pseudobulk_embeddings, celltype_embeddings)
        
        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (pseudobulk_embeddings, celltype_embeddings)), grads = grad_fn(params)
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, pseudobulk_embeddings, celltype_embeddings, loss
    
    except Exception as e:
        logging.error(f"Error in train step: {str(e)}")
        raise

def load_and_preprocess_data(pseudobulk_path, celltype_path, config, tokenizer):
    """Load and preprocess both pseudobulk and celltype-specific data."""
    # Load pseudobulk data
    logging.info("Loading pseudobulk data...")
    pseudobulk_df = pd.read_csv(pseudobulk_path, index_col=0)
    pseudobulk_df = pseudobulk_df.apply(pd.to_numeric, errors='coerce')
    pseudobulk_df = pseudobulk_df.fillna(0)
    
    # Load celltype-specific data
    logging.info("Loading celltype-specific data...")
    celltype_df = pd.read_csv(celltype_path, index_col=0)
    celltype_df = celltype_df.apply(pd.to_numeric, errors='coerce')
    celltype_df = celltype_df.fillna(0)
    
    # Extract donor IDs and convert to numeric indices
    celltype_donors = [idx.split('|')[1] for idx in celltype_df.index]
    pseudobulk_donors = pseudobulk_df.index.tolist()
    
    # Create mapping dictionaries for donor IDs to numeric indices
    unique_donors = list(set(pseudobulk_donors + celltype_donors))
    donor_to_idx = {donor: idx for idx, donor in enumerate(unique_donors)}
    
    # Convert donor IDs to numeric indices
    pseudobulk_donor_indices = jnp.array([donor_to_idx[donor] for donor in pseudobulk_donors], dtype=jnp.int32)
    celltype_donor_indices = jnp.array([donor_to_idx[donor] for donor in celltype_donors], dtype=jnp.int32)
    
    # Preprocess data for the model
    logging.info("Preprocessing data for model...")
    pseudobulk_array = preprocess_rna_seq_for_bulkrnabert(pseudobulk_df, config)
    celltype_array = preprocess_rna_seq_for_bulkrnabert(celltype_df, config)
    
    # Tokenize the data
    logging.info("Tokenizing data...")
    pseudobulk_tokens = jnp.asarray(tokenizer.batch_tokenize(pseudobulk_array), dtype=jnp.int32)
    celltype_tokens = jnp.asarray(tokenizer.batch_tokenize(celltype_array), dtype=jnp.int32)
    
    # Log data shapes and sample counts
    logging.info(f"Pseudobulk data shape: {pseudobulk_df.shape}")
    logging.info(f"Celltype data shape: {celltype_df.shape}")
    logging.info(f"Number of pseudobulk samples: {len(pseudobulk_tokens)}")
    logging.info(f"Number of celltype samples: {len(celltype_tokens)}")
    
    return pseudobulk_tokens, celltype_tokens, pseudobulk_donor_indices, celltype_donor_indices

def create_batches(tokens, batch_size):
    """Create batches from tokens with proper sharding."""
    num_samples = len(tokens)
    # Ensure batch size is divisible by number of devices
    effective_batch_size = (batch_size // NUM_DEVICES) * NUM_DEVICES
    if effective_batch_size == 0:
        effective_batch_size = NUM_DEVICES
    
    for i in range(0, num_samples, effective_batch_size):
        end_idx = min(i + effective_batch_size, num_samples)
        batch = tokens[i:end_idx]
        
        # Pad batch if needed to make it divisible by NUM_DEVICES
        if len(batch) % NUM_DEVICES != 0:
            pad_size = NUM_DEVICES - (len(batch) % NUM_DEVICES)
            padding = [(0, pad_size)] + [(0, 0)] * (len(batch.shape) - 1)
            batch = jnp.pad(batch, padding)
        
        # Reshape batch for sharding
        batch = batch.reshape(NUM_DEVICES, -1, *batch.shape[1:])
        yield batch

def clear_memory():
    """Aggressive memory clearing function."""
    jax.clear_caches()
    for _ in range(3):
        gc.collect()

def train(params, forward_fn, tokenizer, config, num_epochs=50, batch_size=1, learning_rate=1e-4):
    """Main training function."""
    try:
        # Initialize optimizer
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate)
        )
        opt_state = optimizer.init(params)
        
        # Load and preprocess data
        pseudobulk_tokens, celltype_tokens, pseudobulk_donors, celltype_donors = load_and_preprocess_data(
            "data/processed_pseudobulk_expression_W.csv",
            "data/celltype_specific_2d_matrix.csv",
            config,
            tokenizer
        )
        
        # Calculate total number of batches
        num_batches = min(
            (len(pseudobulk_tokens) + batch_size - 1) // batch_size,
            (len(celltype_tokens) + batch_size - 1) // batch_size
        )
        logging.info(f"Number of batches per epoch: {num_batches}")
        
        # Training history
        history = {
            'epoch': [],
            'loss': [],
            'best_loss': []
        }
        
        # Early stopping variables
        best_loss = float('inf')
        best_params = None
        patience = 5  # Number of epochs to wait for improvement
        min_delta = 1e-4  # Minimum change in loss to be considered as improvement
        no_improvement_count = 0
        
        # Training loop
        for epoch in range(num_epochs):
            try:
                epoch_loss = 0
                num_batches_processed = 0
                
                # Process batches
                for i in range(num_batches):
                    try:
                        rng_key = jax.random.PRNGKey(epoch * 1000 + i)
                        
                        # Get current batch
                        batch_start = i * batch_size
                        batch_end = min((i + 1) * batch_size, len(pseudobulk_tokens))
                        
                        # Get batch data
                        pseudobulk_batch = pseudobulk_tokens[batch_start:batch_end]
                        celltype_batch = celltype_tokens[batch_start:batch_end]
                        pseudobulk_batch_donors = pseudobulk_donors[batch_start:batch_end]
                        celltype_batch_donors = celltype_donors[batch_start:batch_end]
                        
                        # Training step
                        params, opt_state, pseudobulk_embeddings, celltype_embeddings, batch_loss = train_step(
                            params, opt_state,
                            pseudobulk_batch,
                            celltype_batch,
                            forward_fn, optimizer, rng_key,
                            pseudobulk_batch_donors,
                            celltype_batch_donors
                        )
                        
                        # Save embeddings
                        np.save(f'embeddings/pseudobulk_embeddings_epoch_{epoch}_batch_{i}.npy', np.array(pseudobulk_embeddings))
                        np.save(f'embeddings/celltype_embeddings_epoch_{epoch}_batch_{i}.npy', np.array(celltype_embeddings))
                        
                        epoch_loss += float(batch_loss)
                        num_batches_processed += 1
                        
                        # Clear memory after each batch
                        clear_memory()
                        
                    except Exception as e:
                        logging.error(f"Error in batch {i} of epoch {epoch + 1}: {str(e)}")
                        continue
                
                # Skip epoch if no successful batches
                if num_batches_processed == 0:
                    logging.error(f"No successful batches in epoch {epoch + 1}")
                    continue
                
                # Compute average epoch loss
                avg_epoch_loss = epoch_loss / num_batches_processed
                
                # Update history
                history['epoch'].append(epoch + 1)
                history['loss'].append(avg_epoch_loss)
                history['best_loss'].append(min(avg_epoch_loss, best_loss))
                
                # Log progress
                logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
                
                # Save best model
                if avg_epoch_loss < (best_loss - min_delta):
                    best_loss = avg_epoch_loss
                    best_params = params
                    checkpoint_path = "checkpoints/best_model.pkl"
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump({
                            'params': params,
                            'loss': avg_epoch_loss
                        }, f)
                    logging.info(f"Saved new best model with loss: {best_loss:.4f}")
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= patience:
                        logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
                
                # Save checkpoints every 5 epochs
                if (epoch + 1) % 5 == 0:
                    checkpoint_path = f"checkpoints/epoch_{epoch + 1}.pkl"
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump({
                            'params': params,
                            'loss': avg_epoch_loss
                        }, f)
                    logging.info(f"Saved checkpoint for epoch {epoch + 1}")
                
                # Clear memory after each epoch
                clear_memory()
                
            except Exception as e:
                logging.error(f"Error in epoch {epoch + 1}: {str(e)}")
                continue
        
        # Save final model
        final_checkpoint = "checkpoints/final_model.pkl"
        with open(final_checkpoint, 'wb') as f:
            pickle.dump({
                'params': best_params,
                'loss': best_loss
            }, f)
        
        history_df = pd.DataFrame(history)
        history_df.to_csv('training_history.csv', index=False)
        
        return best_params, history
        
    except Exception as e:
        logging.error(f"Error in training: {str(e)}")
        raise

def main():
    try:
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
        
        # Enable gradient checkpointing
        config.use_gradient_checkpointing = True
        
        # Train model
        best_params, history = train(
            parameters,
            forward_fn,
            tokenizer,
            config,
            num_epochs=50,
            batch_size=1,
            learning_rate=1e-4
        )
        
        logging.info("Training completed successfully!")
        logging.info(f"Best model saved in checkpoints directory")
        logging.info(f"Training history saved to training_history.csv")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 