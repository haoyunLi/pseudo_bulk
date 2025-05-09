import haiku as hk
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from multiomics_open_research.bulk_rna_bert.pretrained import get_pretrained_model
from multiomics_open_research.bulk_rna_bert.preprocess import preprocess_rna_seq_for_bulkrnabert
from contrastive_loss import compute_contrastive_loss
from evaluation import compute_similarity_metrics, track_training_progress, evaluate_model
import gc
import logging
import os
from tqdm import tqdm
import optax
from sklearn.preprocessing import LabelEncoder
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Configure JAX for distributed training
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'  # Force JAX to see both GPUs
jax.config.update('jax_platform_name', 'cuda')
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.DEFAULT)
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_threefry_partitionable', True)
jax.config.update('jax_enable_custom_prng', True)


# Initialize distributed training
devices = jax.devices()
print(f"Available devices: {devices}")

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
    
    # Extract labels from celltype data index
    logging.info("Extracting labels from celltype data index...")
    labels = []
    for idx in celltype_df.index:
        # The index format is "celltype|donorID", so we split on pipe
        label = idx.split('|')[0]
        labels.append(label)
    
    # Convert labels to numeric if they're categorical
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
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
    logging.info(f"Number of unique labels: {len(np.unique(labels))}")
    
    return pseudobulk_tokens, celltype_tokens, pseudobulk_df.index, celltype_df.index, labels, label_encoder

def create_batches(tokens, batch_size):
    """Create batches from tokens."""
    num_samples = len(tokens)
    for i in range(0, num_samples, batch_size):
        yield tokens[i:i + batch_size]

def clear_memory():
    """Aggressive memory clearing function."""
    # Clear JAX caches
    jax.clear_caches()
    
    # Clear Python garbage collector multiple times
    for _ in range(3):
        gc.collect()

def train_step(params, opt_state, pseudobulk_batch, celltype_batch, forward_fn, optimizer, rng_key):
    """Perform a single training step with distributed processing."""
    def loss_fn(params):
        return compute_contrastive_loss(
            pseudobulk_batch,
            celltype_batch,
            forward_fn,
            params,
            rng_key
        )
    
    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(params)
    
    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss

def train_part(params, opt_state, pseudobulk_tokens, celltype_tokens, forward_fn, optimizer, rng_key, batch_size=1, patience=5, min_delta=0.001):
    """Train on a single part of the data with early stopping and distributed processing."""
    epoch_loss = 0
    num_batches = 0
    
    # Early stopping variables
    best_loss = float('inf')
    best_params = None
    no_improvement_count = 0
    
    # Create batches
    pseudobulk_batches = create_batches(pseudobulk_tokens, batch_size)
    celltype_batches = create_batches(celltype_tokens, batch_size)
    
    # Process batches
    for i, (pseudobulk_batch, celltype_batch) in enumerate(zip(pseudobulk_batches, celltype_batches)):
        # Initialize accumulated gradients
        accumulated_grads = None
        accumulated_loss = 0
        num_grad_steps = 0
        
        # Number of gradient accumulation steps
        grad_accum_steps = 4
        
        # Process the full sequence but accumulate gradients
        for step in range(grad_accum_steps):
            # Split batch across devices
            num_devices = len(devices)
            batch_size_per_device = pseudobulk_batch.shape[0] // num_devices
            
            # Reshape batches for parallel processing
            pseudobulk_batch_reshaped = pseudobulk_batch.reshape(num_devices, batch_size_per_device, -1)
            celltype_batch_reshaped = celltype_batch.reshape(num_devices, batch_size_per_device, -1)
            
            # Compute loss and gradients using distributed processing
            params, opt_state, loss = jax.pmap(
                train_step,
                axis_name='batch',
                in_axes=(None, None, 0, 0, None, None, None),
                out_axes=(None, None, 0)
            )(
                params, opt_state,
                pseudobulk_batch_reshaped,
                celltype_batch_reshaped,
                forward_fn, optimizer, rng_key
            )
            
            accumulated_loss += jax.numpy.mean(loss)  # Average loss across devices
            num_grad_steps += 1
            
            # Clear memory after each accumulation step
            clear_memory()
        
        epoch_loss += accumulated_loss / num_grad_steps
        num_batches += 1
        
        # Clear memory after each batch
        clear_memory()
    
    avg_loss = epoch_loss / num_batches
    
    # Early stopping check
    if avg_loss < best_loss - min_delta:
        best_loss = avg_loss
        best_params = params
        no_improvement_count = 0
    else:
        no_improvement_count += 1
    
    # Clear memory after epoch
    clear_memory()
    
    return params, opt_state, avg_loss, best_params, no_improvement_count

def train_phase(parameters, forward_fn, tokenizer, config, phase_num, learning_rate, num_epochs, batch_size, is_randomized=False):
    """Train a single phase (either mapped or randomized data)."""
    logging.info(f"Starting Phase {phase_num} with learning rate {learning_rate}")
    
    # Initialize optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate)
    )
    opt_state = optimizer.init(parameters)
    
    # Initialize training history
    history = {
        'phase': [],
        'part': [],
        'epoch': [],
        'loss': [],
        'early_stopped': []
    }
    
    # Early stopping parameters
    patience = 5
    min_delta = 0.001
    
    # Train each part sequentially
    for part_idx in range(1, 4):  # 3 parts
        logging.info(f"Training on part {part_idx}/3")
        
        # Load and preprocess part data
        pseudobulk_df, celltype_df, labels, label_encoder = load_part_data(part_idx, is_randomized)
        
        # Preprocess data for the model
        pseudobulk_array = preprocess_rna_seq_for_bulkrnabert(pseudobulk_df, config)
        celltype_array = preprocess_rna_seq_for_bulkrnabert(celltype_df, config)
        
        # Tokenize the data
        pseudobulk_tokens = jnp.asarray(tokenizer.batch_tokenize(pseudobulk_array), dtype=jnp.int32)
        celltype_tokens = jnp.asarray(tokenizer.batch_tokenize(celltype_array), dtype=jnp.int32)
        
        # Clear memory after preprocessing
        del pseudobulk_df
        del celltype_df
        del pseudobulk_array
        del celltype_array
        clear_memory()
        
        # Early stopping variables for this part
        best_params_part = parameters
        no_improvement_count = 0
        
        # Train on this part for num_epochs
        for epoch in range(num_epochs):
            rng_key = jax.random.PRNGKey(epoch * 1000 + part_idx)
            parameters, opt_state, epoch_loss, current_best_params, no_improvement = train_part(
                parameters, opt_state,
                pseudobulk_tokens,
                celltype_tokens,
                forward_fn, optimizer, rng_key, batch_size,
                patience, min_delta
            )
            
            # Update best parameters if improved
            if current_best_params is not None:
                best_params_part = current_best_params
            
            # Update early stopping counter
            no_improvement_count = no_improvement
            
            # Update history
            history['phase'].append(phase_num)
            history['part'].append(part_idx)
            history['epoch'].append(epoch + 1)
            history['loss'].append(epoch_loss)
            history['early_stopped'].append(False)
            
            # Log epoch statistics
            logging.info(f"Phase {phase_num} - Part {part_idx} - Epoch {epoch + 1}/{num_epochs}")
            logging.info(f"Loss: {epoch_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f"checkpoints/phase{phase_num}_part{part_idx}_epoch_{epoch + 1}.pkl"
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(parameters, f)
                logging.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Early stopping check
            if no_improvement_count >= patience:
                logging.info(f"Early stopping triggered for part {part_idx} after {epoch + 1} epochs")
                history['early_stopped'][-1] = True
                parameters = best_params_part  # Use best parameters
                break
            
            # Clear memory after each epoch
            clear_memory()
        
        # Clear memory after each part
        del pseudobulk_tokens
        del celltype_tokens
        clear_memory()
    
    return parameters, history

def main():
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Get pretrained model
        logging.info("Loading pretrained model...")
        parameters, forward_fn, tokenizer, config = get_pretrained_model(
            model_name="bulk_rna_bert_gtex_encode",
            embeddings_layers_to_save=(4,),
            checkpoint_directory="multiomics-open-research/checkpoints/",
            compute_dtype=jnp.bfloat16,  # Use bfloat16 for computations (more stable than float16)
            param_dtype=jnp.float32,     # Keep parameters in float32
            output_dtype=jnp.float32     # Keep outputs in float32
        )
        forward_fn = hk.transform(forward_fn)

        # Enable gradient checkpointing
        config.use_gradient_checkpointing = True
        
        # Training parameters
        batch_size = 1
        num_epochs = 50
        high_lr = 1e-3  # Higher learning rate for mapped data
        low_lr = 1e-5   # Lower learning rate for randomized data
        
        # Phase 1: Train on mapped data with high learning rate
        parameters, history_phase1 = train_phase(
            parameters, forward_fn, tokenizer, config,
            phase_num=1,
            learning_rate=high_lr,
            num_epochs=num_epochs,
            batch_size=batch_size,
            is_randomized=False
        )
        
        # Clear memory between phases
        clear_memory()
        
        # Save best model from phase 1
        checkpoint_path = "checkpoints/best_model_phase1.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(parameters, f)
        logging.info("Saved best model from phase 1")
        
        # Phase 2: Train on randomized data with low learning rate
        parameters, history_phase2 = train_phase(
            parameters, forward_fn, tokenizer, config,
            phase_num=2,
            learning_rate=low_lr,
            num_epochs=num_epochs,
            batch_size=batch_size,
            is_randomized=True
        )
        
        # Clear memory before saving final model
        clear_memory()
        
        # Save final model
        checkpoint_path = "checkpoints/final_model.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(parameters, f)
        logging.info("Saved final model")
        
        # Combine and save training history
        history = pd.concat([
            pd.DataFrame(history_phase1),
            pd.DataFrame(history_phase2)
        ])
        history.to_csv('training_history.csv', index=False)
        logging.info("Saved training history")
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 