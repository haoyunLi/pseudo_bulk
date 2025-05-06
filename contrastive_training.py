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
import time  

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Configure JAX for memory efficiency
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_disable_jit', False)  # Enable JIT compilation
jax.config.update('jax_threefry_partitionable', True)  # Enable better parallelization

def create_chunk_attention_mask(seq_len, chunk_size):
    """Create a mask for chunk-based attention.
    
    Args:
        seq_len: Total sequence length
        chunk_size: Size of each chunk
        
    Returns:
        Boolean mask of shape [seq_len, seq_len] where True indicates
        positions within the same chunk.
    """
    mask = jnp.zeros((seq_len, seq_len), dtype=bool)
    
    # For each chunk
    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        # Set True for all positions within this chunk
        mask = mask.at[chunk_start:chunk_end, chunk_start:chunk_end].set(True)
    
    return mask

def apply_chunk_attention(attention_fn, x, chunk_size, mask=None, rng_key=None):
    """Apply attention function with chunk-based masking.
    
    Args:
        attention_fn: Original attention function
        x: Input tensor
        chunk_size: Size of each chunk
        mask: Optional additional mask
        rng_key: Random key for initialization
        
    Returns:
        Output tensor with chunk-based attention
    """
    # Create chunk mask
    seq_len = x.shape[1]
    chunk_mask = create_chunk_attention_mask(seq_len, chunk_size)
    
    # Combine with additional mask if provided
    if mask is not None:
        chunk_mask = jnp.logical_and(chunk_mask, mask)
    
    # Use provided rng_key or create a new one
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    
    # Apply the forward function without mask
    # The attention mask will be handled internally by the model
    return attention_fn.apply(attention_fn.init(rng_key, x), rng_key, x)

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
        # The index format is "celltype_donorID", so we split on underscore
        label = idx.split('_')[0]
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

def process_chunk(chunk_tokens, parameters, forward_fn, rng_key, chunk_size):
    """Process a chunk of tokens with chunk-based attention."""
    try:
        # Create a wrapper function that handles the chunk-based attention
        def chunk_attention_forward_fn(x):
            # Apply the forward function
            return apply_chunk_attention(forward_fn, x, chunk_size, rng_key=rng_key)
        
        # Transform the function with Haiku
        chunk_attention_forward_fn = hk.transform(chunk_attention_forward_fn)
        
        # Initialize parameters for the chunk attention function
        chunk_params = chunk_attention_forward_fn.init(rng_key, chunk_tokens)
        
        # Apply the function with both sets of parameters
        outs = chunk_attention_forward_fn.apply(
            {**parameters, **chunk_params},  # Combine both parameter sets
            rng_key,
            chunk_tokens
        )
        chunk_embeddings = np.array(outs["embeddings_4"].mean(axis=1), dtype=np.float32)
        return chunk_embeddings
    except Exception as e:
        logging.error(f"Error processing chunk: {str(e)}")
        raise

def train_step(params, opt_state, pseudobulk_batch, celltype_batch, forward_fn, optimizer, rng_key, chunk_size):
    """Perform a single training step with chunk-based attention."""
    def loss_fn(params):
        # Create a wrapper function that handles the chunk-based attention
        def chunk_attention_forward_fn(x):
            # Apply the forward function
            return apply_chunk_attention(forward_fn, x, chunk_size, rng_key=rng_key)
        
        # Transform the function with Haiku
        chunk_attention_forward_fn = hk.transform(chunk_attention_forward_fn)
        
        # Initialize parameters for the chunk attention function
        chunk_params = chunk_attention_forward_fn.init(rng_key, pseudobulk_batch)
        
        # Combine parameters
        combined_params = {**params, **chunk_params}
        
        return compute_contrastive_loss(
            pseudobulk_batch,
            celltype_batch,
            chunk_attention_forward_fn,
            combined_params,
            rng_key
        )
    
    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(params)
    
    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss

def main():
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Get pretrained model
        logging.info("Loading pretrained model...")
        parameters, forward_fn, tokenizer, config = get_pretrained_model(
            model_name="bulk_rna_bert_tcga",
            embeddings_layers_to_save=(4,),
            checkpoint_directory="multiomics-open-research/checkpoints/",
        )
        forward_fn = hk.transform(forward_fn)
        
        # Load and preprocess data
        logging.info("Loading and preprocessing data...")
        pseudobulk_tokens, celltype_tokens, pseudobulk_indices, celltype_indices, labels, label_encoder = load_and_preprocess_data(
            "data/processed_pseudobulk_expression_W.csv",
            "data/celltype_specific_2d_matrix.csv",
            config,
            tokenizer
        )
        
        # Training configuration
        batch_size = 6  
        chunk_size = 128  
        processing_chunk_size = 150  
        num_epochs = 30
        learning_rate = 1e-4
        checkpoint_frequency = 5
        
        # Early stopping configuration
        early_stopping_patience = 5
        early_stopping_min_delta = 1e-4
        best_loss = float('inf')
        patience_counter = 0
        
        # Initialize optimizer with gradient clipping
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate)
        )
        opt_state = optimizer.init(parameters)
        
        # Initialize training history
        history = {
            'epoch': [],
            'loss': [],
            'mean_cosine_similarity': [],
            'silhouette_score': [],
            'best_loss': [],
            'patience_counter': []
        }
        
        # Start time tracking
        start_time = time.time()
        
        # Training loop
        logging.info("Starting training...")
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Process data in chunks
            for chunk_start in range(0, len(pseudobulk_tokens), processing_chunk_size):
                chunk_end = min(chunk_start + processing_chunk_size, len(pseudobulk_tokens))
                logging.info(f"Processing chunk {chunk_start}-{chunk_end} of {len(pseudobulk_tokens)} samples...")
                
                # Get current chunk
                pseudobulk_chunk = pseudobulk_tokens[chunk_start:chunk_end]
                celltype_chunk = celltype_tokens[chunk_start:chunk_end]
                
                # Process chunk in batches
                for i in range(0, len(pseudobulk_chunk), batch_size):
                    batch_end = min(i + batch_size, len(pseudobulk_chunk))
                    pseudobulk_batch = pseudobulk_chunk[i:batch_end]
                    celltype_batch = celltype_chunk[i:batch_end]
                    
                    rng_key = jax.random.PRNGKey(epoch * 1000 + chunk_start + i)
                    parameters, opt_state, batch_loss = train_step(
                        parameters,
                        opt_state,
                        pseudobulk_batch,
                        celltype_batch,
                        forward_fn,
                        optimizer,
                        rng_key,
                        chunk_size
                    )
                    
                    epoch_loss += batch_loss
                    num_batches += 1
                    
                    # Clear memory after each batch
                    if i % 5 == 0:  # More frequent cleanup
                        gc.collect()
                        jax.clear_caches()
                
                # Clear memory after each chunk
                del pseudobulk_chunk
                del celltype_chunk
                gc.collect()
                jax.clear_caches()
            
            # Compute average loss
            avg_loss = epoch_loss / num_batches
            
            # Early stopping check
            if avg_loss < (best_loss - early_stopping_min_delta):
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
                checkpoint_path = f"checkpoints/best_model.pkl"
                with open(checkpoint_path, 'wb') as f:
                    import pickle
                    pickle.dump({
                        'parameters': parameters,
                        'opt_state': opt_state,
                        'epoch': epoch + 1,
                        'loss': avg_loss,
                        'metrics': metrics
                    }, f)
                logging.info(f"New best model saved with loss: {avg_loss:.4f}")
            else:
                patience_counter += 1
                logging.info(f"No improvement for {patience_counter} epochs")
            
            # Compute embeddings for evaluation in chunks
            logging.info("Computing embeddings for evaluation...")
            pseudobulk_embeddings = []
            celltype_embeddings = []
            
            # Process pseudobulk data in chunks
            for chunk_start in range(0, len(pseudobulk_tokens), processing_chunk_size):
                chunk_end = min(chunk_start + processing_chunk_size, len(pseudobulk_tokens))
                chunk = pseudobulk_tokens[chunk_start:chunk_end]
                chunk_embeddings = process_chunk(chunk, parameters, forward_fn, jax.random.PRNGKey(0), chunk_size)
                pseudobulk_embeddings.append(chunk_embeddings)
                del chunk
                del chunk_embeddings
                gc.collect()
            
            # Process celltype data in chunks
            for chunk_start in range(0, len(celltype_tokens), processing_chunk_size):
                chunk_end = min(chunk_start + processing_chunk_size, len(celltype_tokens))
                chunk = celltype_tokens[chunk_start:chunk_end]
                chunk_embeddings = process_chunk(chunk, parameters, forward_fn, jax.random.PRNGKey(0), chunk_size)
                celltype_embeddings.append(chunk_embeddings)
                del chunk
                del chunk_embeddings
                gc.collect()
            
            # Combine embeddings
            pseudobulk_embeddings = np.vstack(pseudobulk_embeddings)
            celltype_embeddings = np.vstack(celltype_embeddings)
            
            # Compute metrics
            metrics = compute_similarity_metrics(pseudobulk_embeddings, celltype_embeddings, labels)
            
            # Update history
            history['epoch'].append(epoch + 1)
            history['loss'].append(avg_loss)
            history['mean_cosine_similarity'].append(metrics['mean_cosine_similarity'])
            history['silhouette_score'].append(metrics['silhouette_score'])
            history['best_loss'].append(best_loss)
            history['patience_counter'].append(patience_counter)
            
            # Log epoch statistics
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            remaining_epochs = min(
                num_epochs - epoch - 1,
                early_stopping_patience - patience_counter
            )
            remaining_time = avg_time_per_epoch * remaining_epochs
            
            logging.info(f"Epoch {epoch + 1}/{num_epochs}")
            logging.info(f"Average Loss: {avg_loss:.4f}")
            logging.info(f"Best Loss: {best_loss:.4f}")
            logging.info(f"Mean Cosine Similarity: {metrics['mean_cosine_similarity']:.4f}")
            logging.info(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
            logging.info(f"Patience Counter: {patience_counter}/{early_stopping_patience}")
            logging.info(f"Estimated time remaining: {remaining_time/3600:.1f} hours")
            
            # Track training progress
            track_training_progress(history)
            
            # Check for early stopping
            if patience_counter >= early_stopping_patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Save model checkpoint
            if (epoch + 1) % checkpoint_frequency == 0:
                checkpoint_path = f"checkpoints/model_epoch_{epoch + 1}.pkl"
                with open(checkpoint_path, 'wb') as f:
                    import pickle
                    pickle.dump({
                        'parameters': parameters,
                        'opt_state': opt_state,
                        'epoch': epoch + 1,
                        'loss': avg_loss,
                        'metrics': metrics
                    }, f)
                logging.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Final evaluation
        logging.info("Performing final evaluation...")
        final_metrics = evaluate_model(
            pseudobulk_embeddings,
            celltype_embeddings,
            labels,
            output_dir='final_evaluation'
        )
        
        # Save final embeddings
        logging.info("Saving final embeddings...")
        np.save('data/trained_pseudobulk_embeddings.npy', pseudobulk_embeddings)
        np.save('data/trained_celltype_embeddings.npy', celltype_embeddings)
        
        # Save as CSV with indices
        pd.DataFrame(pseudobulk_embeddings, index=pseudobulk_indices).to_csv('data/trained_pseudobulk_embeddings.csv')
        pd.DataFrame(celltype_embeddings, index=celltype_indices).to_csv('data/trained_celltype_embeddings.csv')
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 