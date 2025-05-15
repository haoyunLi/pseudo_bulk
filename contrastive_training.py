import haiku as hk
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from multiomics_open_research.bulk_rna_bert.pretrained import get_pretrained_model
from multiomics_open_research.bulk_rna_bert.preprocess import preprocess_rna_seq_for_bulkrnabert
from contrastive_loss import compute_contrastive_loss
import gc
import logging
import os
import optax
import pickle
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh
from jax.experimental.pjit import pjit

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
batch_spec = P('model')  # Batch dimension is sharded across model dimension

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
    
    # Extract donor IDs from celltype data index
    celltype_donors = [idx.split('|')[1] for idx in celltype_df.index]
    pseudobulk_donors = pseudobulk_df.index.tolist()
    
    # Preprocess data for the model
    logging.info("Preprocessing data for model...")
    pseudobulk_array = preprocess_rna_seq_for_bulkrnabert(pseudobulk_df, config)
    celltype_array = preprocess_rna_seq_for_bulkrnabert(celltype_df, config)
    
    # Tokenize the data
    logging.info("Tokenizing data...")
    pseudobulk_tokens = jnp.asarray(tokenizer.batch_tokenize(pseudobulk_array), dtype=jnp.int32)
    celltype_tokens = jnp.asarray(tokenizer.batch_tokenize(celltype_array), dtype=jnp.int32)
    
    # Log data shapes
    logging.info(f"Pseudobulk data shape: {pseudobulk_df.shape}")
    logging.info(f"Celltype data shape: {celltype_df.shape}")
    
    return pseudobulk_tokens, celltype_tokens, pseudobulk_donors, celltype_donors

def create_batches(tokens, batch_size):
    """Create batches from tokens."""
    num_samples = len(tokens)
    for i in range(0, num_samples, batch_size):
        yield tokens[i:i + batch_size]

def clear_memory():
    """Aggressive memory clearing function."""
    jax.clear_caches()
    for _ in range(3):
        gc.collect()

def train_step(params, opt_state, pseudobulk_batch, celltype_batch, forward_fn, optimizer, rng_key, pseudobulk_donors, celltype_donors, is_pseudobulk_phase=True):
    """Single training step computing contrastive loss for both modalities."""
    # Create a closure that captures forward_fn
    def create_sharded_compute(forward_fn):
        def sharded_compute(params, opt_state, pseudobulk_batch, celltype_batch, optimizer, rng_key, pseudobulk_donors, celltype_donors, is_pseudobulk_phase):
            # Generate embeddings for both batches
            pseudobulk_outs = forward_fn.apply(params, rng_key, pseudobulk_batch)
            celltype_outs = forward_fn.apply(params, rng_key, celltype_batch)
            
            pseudobulk_embeddings = pseudobulk_outs["embeddings_4"].mean(axis=1)
            celltype_embeddings = celltype_outs["embeddings_4"].mean(axis=1)
            
            def loss_fn(params):
                # Forward pass for both batches
                pseudobulk_outs = forward_fn.apply(params, rng_key, pseudobulk_batch)
                celltype_outs = forward_fn.apply(params, rng_key, celltype_batch)
                
                pseudobulk_embeddings = pseudobulk_outs["embeddings_4"].mean(axis=1)
                celltype_embeddings = celltype_outs["embeddings_4"].mean(axis=1)
                
                # Compute contrastive loss in both directions
                pseudobulk_loss, celltype_loss = compute_contrastive_loss(
                    pseudobulk_embeddings,
                    celltype_embeddings,
                    pseudobulk_donors,
                    celltype_donors,
                    is_training=True
                )
                
                # Focus on the current phase's loss
                if is_pseudobulk_phase:
                    return pseudobulk_loss
                else:
                    return celltype_loss
            
            # Compute gradients
            grad_fn = jax.value_and_grad(loss_fn)
            (total_loss, grads) = grad_fn(params)
            
            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            return params, opt_state, pseudobulk_embeddings, celltype_embeddings, total_loss
        
        return sharded_compute
    
    # Create the sharded computation function
    sharded_compute = create_sharded_compute(forward_fn)
    
    # Create pjit function with sharding specs
    sharded_train_step = pjit(
        sharded_compute,
        in_axis_resources=(param_spec, None, batch_spec, batch_spec, None, None, None, None, None),
        out_axis_resources=(param_spec, None, batch_spec, batch_spec, None)
    )
    
    # Execute sharded computation
    with mesh:
        params, opt_state, pseudobulk_embeddings, celltype_embeddings, total_loss = sharded_train_step(
            params, opt_state,
            pseudobulk_batch,
            celltype_batch,
            optimizer, rng_key,
            pseudobulk_donors,
            celltype_donors,
            is_pseudobulk_phase
        )
    
    return params, opt_state, pseudobulk_embeddings, celltype_embeddings, total_loss

def train(params, forward_fn, tokenizer, config, num_epochs=50, batch_size=1, learning_rate=1e-4):
    """Main training function."""
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
    
    # Training history
    history = {
        'epoch': [],
        'phase': [],
        'loss': [],
        'pseudobulk_loss': [],
        'celltype_loss': [],
        'best_pseudobulk_loss': [],
        'best_celltype_loss': []
    }
    
    # Early stopping variables
    best_pseudobulk_loss = float('inf')
    best_celltype_loss = float('inf')
    best_pseudobulk_params = None
    best_celltype_params = None
    patience = 5  # Number of epochs to wait for improvement
    min_delta = 1e-4  # Minimum change in loss to be considered as improvement
    no_improvement_count = 0
    
    # Create batches
    pseudobulk_batches = create_batches(pseudobulk_tokens, batch_size)
    celltype_batches = create_batches(celltype_tokens, batch_size)
    
    # Training loop
    for epoch in range(num_epochs):
        # Alternate between pseudobulk and celltype phases
        for phase in ['pseudobulk', 'celltype']:
            is_pseudobulk_phase = (phase == 'pseudobulk')
            epoch_loss = 0
            epoch_pseudobulk_loss = 0
            epoch_celltype_loss = 0
            num_batches = 0
            
            # Process batches
            for i, (pseudobulk_batch, celltype_batch) in enumerate(zip(pseudobulk_batches, celltype_batches)):
                rng_key = jax.random.PRNGKey(epoch * 1000 + i)
                
                # Training step
                params, opt_state, pseudobulk_embeddings, celltype_embeddings, batch_loss = train_step(
                    params, opt_state,
                    pseudobulk_batch,
                    celltype_batch,
                    forward_fn, optimizer, rng_key,
                    pseudobulk_donors[i:i + batch_size],
                    celltype_donors[i:i + batch_size],
                    is_pseudobulk_phase
                )
                
                # Save embeddings
                np.save(f'embeddings/pseudobulk_embeddings_epoch_{epoch}_phase_{phase}_batch_{i}.npy', np.array(pseudobulk_embeddings))
                np.save(f'embeddings/celltype_embeddings_epoch_{epoch}_phase_{phase}_batch_{i}.npy', np.array(celltype_embeddings))
                
                # Compute separate losses for pseudobulk and celltype
                pseudobulk_loss, celltype_loss = compute_contrastive_loss(
                    pseudobulk_embeddings,
                    celltype_embeddings,
                    pseudobulk_donors[i:i + batch_size],
                    celltype_donors[i:i + batch_size],
                    is_training=False
                )
                
                epoch_loss += float(batch_loss)
                epoch_pseudobulk_loss += float(pseudobulk_loss)
                epoch_celltype_loss += float(celltype_loss)
                num_batches += 1
                
                # Clear memory
                clear_memory()
            
            # Compute average epoch losses
            avg_epoch_loss = epoch_loss / num_batches
            avg_pseudobulk_loss = epoch_pseudobulk_loss / num_batches
            avg_celltype_loss = epoch_celltype_loss / num_batches
            
            # Update history
            history['epoch'].append(epoch + 1)
            history['phase'].append(phase)
            history['loss'].append(avg_epoch_loss)
            history['pseudobulk_loss'].append(avg_pseudobulk_loss)
            history['celltype_loss'].append(avg_celltype_loss)
            history['best_pseudobulk_loss'].append(min(avg_pseudobulk_loss, best_pseudobulk_loss))
            history['best_celltype_loss'].append(min(avg_celltype_loss, best_celltype_loss))
            
            # Log progress
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Phase: {phase}")
            logging.info(f"Total Loss: {avg_epoch_loss:.4f}, Pseudobulk Loss: {avg_pseudobulk_loss:.4f}, Celltype Loss: {avg_celltype_loss:.4f}")
            
            # Save best models for each phase
            if is_pseudobulk_phase and avg_pseudobulk_loss < (best_pseudobulk_loss - min_delta):
                best_pseudobulk_loss = avg_pseudobulk_loss
                best_pseudobulk_params = params
                pseudobulk_checkpoint = "checkpoints/best_pseudobulk_model.pkl"
                with open(pseudobulk_checkpoint, 'wb') as f:
                    pickle.dump({
                        'params': params,
                        'loss': avg_pseudobulk_loss
                    }, f)
                logging.info(f"Saved new best pseudobulk model with loss: {best_pseudobulk_loss:.4f}")
                no_improvement_count = 0
            elif not is_pseudobulk_phase and avg_celltype_loss < (best_celltype_loss - min_delta):
                best_celltype_loss = avg_celltype_loss
                best_celltype_params = params
                celltype_checkpoint = "checkpoints/best_celltype_model.pkl"
                with open(celltype_checkpoint, 'wb') as f:
                    pickle.dump({
                        'params': params,
                        'loss': avg_celltype_loss
                    }, f)
                logging.info(f"Saved new best celltype model with loss: {best_celltype_loss:.4f}")
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Save checkpoints every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f"checkpoints/epoch_{epoch + 1}_phase_{phase}.pkl"
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump({
                        'params': params,
                        'pseudobulk_loss': avg_pseudobulk_loss,
                        'celltype_loss': avg_celltype_loss
                    }, f)
                logging.info(f"Saved checkpoint for epoch {epoch + 1}, phase {phase}")
        
        if no_improvement_count >= patience:
            break
    
    # Save final models
    final_checkpoint = "checkpoints/final_model.pkl"
    with open(final_checkpoint, 'wb') as f:
        pickle.dump({
            'pseudobulk_params': best_pseudobulk_params,
            'celltype_params': best_celltype_params,
            'pseudobulk_loss': best_pseudobulk_loss,
            'celltype_loss': best_celltype_loss
        }, f)
    
    history_df = pd.DataFrame(history)
    history_df.to_csv('training_history.csv', index=False)
    
    return best_pseudobulk_params, best_celltype_params, history

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
        best_pseudobulk_params, best_celltype_params, history = train(
            parameters,
            forward_fn,
            tokenizer,
            config,
            num_epochs=50,
            batch_size=1,
            learning_rate=1e-4
        )
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 