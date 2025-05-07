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
import glob
import json
from sklearn.metrics import silhouette_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Configure JAX for memory efficiency
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.DEFAULT)
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_disable_jit', False)  # Enable JIT compilation
jax.config.update('jax_threefry_partitionable', True)  # Enable better parallelization

# Configure GPU memory growth
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'  # Use 70% of available GPU memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Don't preallocate all GPU memory

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
    
    # Create output directories for chunks
    os.makedirs('data/chunks/pseudobulk', exist_ok=True)
    os.makedirs('data/chunks/celltype', exist_ok=True)
    
    # Preprocess data for the model in chunks
    logging.info("Preprocessing data for model...")
    chunk_size = 1000  # Process 1000 samples at a time
    
    # Process pseudobulk data in chunks
    num_pseudobulk_chunks = (len(pseudobulk_df) + chunk_size - 1) // chunk_size
    for i in range(num_pseudobulk_chunks):
        chunk = pseudobulk_df.iloc[i*chunk_size:(i+1)*chunk_size]
        chunk_array = preprocess_rna_seq_for_bulkrnabert(chunk, config)
        chunk_tokens = jnp.asarray(tokenizer.batch_tokenize(chunk_array), dtype=jnp.int32)
        
        # Save chunk to disk
        np.save(f'data/chunks/pseudobulk/chunk_{i}.npy', np.array(chunk_tokens))
        del chunk_array
        del chunk_tokens
        gc.collect()
        jax.clear_caches()
    
    # Process celltype data in chunks
    num_celltype_chunks = (len(celltype_df) + chunk_size - 1) // chunk_size
    for i in range(num_celltype_chunks):
        chunk = celltype_df.iloc[i*chunk_size:(i+1)*chunk_size]
        chunk_array = preprocess_rna_seq_for_bulkrnabert(chunk, config)
        chunk_tokens = jnp.asarray(tokenizer.batch_tokenize(chunk_array), dtype=jnp.int32)
        
        # Save chunk to disk
        np.save(f'data/chunks/celltype/chunk_{i}.npy', np.array(chunk_tokens))
        del chunk_array
        del chunk_tokens
        gc.collect()
        jax.clear_caches()
    
    # Log data shapes
    logging.info(f"Pseudobulk data shape: {pseudobulk_df.shape}")
    logging.info(f"Celltype data shape: {celltype_df.shape}")
    logging.info(f"Number of pseudobulk chunks: {num_pseudobulk_chunks}")
    logging.info(f"Number of celltype chunks: {num_celltype_chunks}")
    logging.info(f"Number of unique labels: {len(np.unique(labels))}")
    
    return pseudobulk_df.index, celltype_df.index, labels, label_encoder, num_pseudobulk_chunks, num_celltype_chunks

def process_batch(batch_tokens, parameters, forward_fn, rng_key):
    """Process a batch of tokens with regular attention."""
    try:
        # Apply the forward function
        outs = forward_fn.apply(parameters, rng_key, batch_tokens)
        
        # Get embeddings and mean pool
        batch_embeddings = np.array(outs["embeddings_4"].mean(axis=1), dtype=np.float32)
        
        return batch_embeddings
        
    except Exception as e:
        logging.error(f"Error processing batch: {str(e)}")
        raise

def train_step(params, opt_state, pseudobulk_batch, celltype_batch, forward_fn, optimizer, rng_key, grad_accum_steps=1):
    """Perform a single training step with gradient accumulation."""
    def loss_fn(params):
        # Process celltype data in smaller chunks
        chunk_size = 100  # Process 100 celltype samples at a time
        total_loss = 0.0
        
        for i in range(0, len(celltype_batch), chunk_size):
            chunk_end = min(i + chunk_size, len(celltype_batch))
            celltype_chunk = celltype_batch[i:chunk_end]
            
            # Compute loss for this chunk
            chunk_loss = compute_contrastive_loss(
                pseudobulk_batch,
                celltype_chunk,
                forward_fn,
                params,
                rng_key
            )
            total_loss += chunk_loss * (chunk_end - i) / len(celltype_batch)
            
            # Clear memory after each chunk
            gc.collect()
            jax.clear_caches()
        
        return total_loss
    
    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(params)
    
    # Scale gradients by accumulation steps
    grads = jax.tree_map(lambda x: x / grad_accum_steps, grads)
    
    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss

def save_checkpoint(parameters, opt_state, epoch, loss, metrics, path):
    """Save model checkpoint with metrics."""
    with open(path, 'wb') as f:
        import pickle
        pickle.dump({
            'parameters': parameters,
            'opt_state': opt_state,
            'epoch': epoch,
            'loss': loss,
            'metrics': metrics
        }, f)
    logging.info(f"Saved checkpoint to {path}")

def process_batch_from_chunks(chunk_idx, batch_idx, batch_size, chunk_dir):
    """Load a specific batch from a chunk file."""
    chunk = np.load(f'{chunk_dir}/chunk_{chunk_idx}.npy')
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(chunk))
    batch = chunk[start_idx:end_idx]
    return batch

def compute_metrics_streaming(embeddings_dir, num_chunks, labels, chunk_size=1000):
    """Compute metrics in a streaming way without loading all embeddings at once."""
    # Initialize metrics accumulators
    total_cosine_sim = 0.0
    total_samples = 0
    
    # Process chunks in batches
    for chunk_idx in range(num_chunks):
        # Load chunk embeddings
        chunk_embeddings = np.load(f'{embeddings_dir}/chunk_{chunk_idx}.npy')
        
        # Process chunk in smaller batches
        for i in range(0, len(chunk_embeddings), chunk_size):
            batch_end = min(i + chunk_size, len(chunk_embeddings))
            batch_embeddings = chunk_embeddings[i:batch_end]
            batch_labels = labels[i:batch_end]
            
            # Compute cosine similarity for this batch
            batch_cosine_sim = np.mean(np.dot(batch_embeddings, batch_embeddings.T))
            total_cosine_sim += batch_cosine_sim * len(batch_embeddings)
            total_samples += len(batch_embeddings)
            
            # Clear memory
            del batch_embeddings
            gc.collect()
            jax.clear_caches()
        
        # Clear chunk memory
        del chunk_embeddings
        gc.collect()
        jax.clear_caches()
    
    # Compute final metrics
    mean_cosine_similarity = total_cosine_sim / total_samples if total_samples > 0 else 0.0
    
    return {
        'mean_cosine_similarity': mean_cosine_similarity,
        'silhouette_score': 0.0  # Silhouette score requires all data, so we'll skip it
    }

def save_embeddings_memmap(chunk_dir, output_path, expected_shape, dtype=np.float32):
    """Save embeddings using memory-mapped arrays, handling multiple chunk files."""
    # Get all chunk files and sort them
    chunk_files = sorted(glob.glob(f'{chunk_dir}/chunk_*.npy'))
    if not chunk_files:
        raise ValueError(f"No chunk files found in {chunk_dir}")
    
    # Load first chunk to get embedding dimension
    first_chunk = np.load(chunk_files[0])
    embedding_dim = first_chunk.shape[1]
    del first_chunk
    gc.collect()
    
    # Create memory-mapped array with correct shape
    total_samples = expected_shape[0]
    fp = np.memmap(output_path, dtype=dtype, mode='w+', shape=(total_samples, embedding_dim))
    
    # Write data in chunks
    current_idx = 0
    for chunk_file in chunk_files:
        chunk = np.load(chunk_file)
        chunk_size = len(chunk)
        fp[current_idx:current_idx + chunk_size] = chunk
        current_idx += chunk_size
        del chunk
        gc.collect()
    
    # Flush changes to disk
    fp.flush()
    del fp

def compute_approximate_silhouette(embeddings_dir, num_chunks, labels, sample_size=1000):
    """Compute approximate silhouette score using subsampling."""
    # Get all chunk files
    chunk_files = sorted(glob.glob(f'{embeddings_dir}/chunk_*.npy'))
    if not chunk_files:
        return 0.0
    
    # Randomly sample indices
    total_samples = sum(len(np.load(f)) for f in chunk_files)
    sample_indices = np.random.choice(total_samples, min(sample_size, total_samples), replace=False)
    sample_indices.sort()  # Sort for efficient loading
    
    # Load sampled data
    sampled_embeddings = []
    sampled_labels = []
    current_idx = 0
    
    for chunk_file in chunk_files:
        chunk = np.load(chunk_file)
        chunk_size = len(chunk)
        
        # Find indices that fall in this chunk
        chunk_indices = sample_indices[(sample_indices >= current_idx) & 
                                     (sample_indices < current_idx + chunk_size)]
        if len(chunk_indices) > 0:
            # Convert to local indices
            local_indices = chunk_indices - current_idx
            sampled_embeddings.append(chunk[local_indices])
            sampled_labels.append(labels[chunk_indices])
        
        current_idx += chunk_size
        del chunk
        gc.collect()
    
    if not sampled_embeddings:
        return 0.0
    
    # Combine sampled data
    sampled_embeddings = np.vstack(sampled_embeddings)
    sampled_labels = np.concatenate(sampled_labels)
    
    # Compute silhouette score on sampled data
    try:
        score = silhouette_score(sampled_embeddings, sampled_labels)
    except Exception as e:
        logging.warning(f"Error computing silhouette score: {str(e)}")
        score = 0.0
    
    # Clean up
    del sampled_embeddings
    del sampled_labels
    gc.collect()
    
    return score

def evaluate_model_streaming(embeddings_dir, num_chunks, labels, output_dir='final_evaluation'):
    """Evaluate model performance in a streaming way."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metrics
    metrics = {
        'mean_cosine_similarity': 0.0,
        'silhouette_score': 0.0
    }
    
    # Compute metrics in streaming fashion
    total_cosine_sim = 0.0
    total_samples = 0
    
    # Process each chunk
    for chunk_idx in range(num_chunks):
        chunk_path = f'{embeddings_dir}/chunk_{chunk_idx}.npy'
        if not os.path.exists(chunk_path):
            continue
            
        # Load chunk
        chunk = np.load(chunk_path)
        chunk_size = len(chunk)
        
        # Compute cosine similarity for this chunk
        chunk_cosine_sim = np.mean(np.dot(chunk, chunk.T))
        total_cosine_sim += chunk_cosine_sim * chunk_size
        total_samples += chunk_size
        
        # Clear memory
        del chunk
        gc.collect()
        jax.clear_caches()
    
    # Compute final metrics
    if total_samples > 0:
        metrics['mean_cosine_similarity'] = total_cosine_sim / total_samples
    
    # Compute approximate silhouette score
    metrics['silhouette_score'] = compute_approximate_silhouette(
        embeddings_dir, num_chunks, labels
    )
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

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
        pseudobulk_indices, celltype_indices, labels, label_encoder, num_pseudobulk_chunks, num_celltype_chunks = load_and_preprocess_data(
            "data/processed_pseudobulk_expression_W.csv",
            "data/celltype_specific_2d_matrix.csv",
            config,
            tokenizer
        )
        
        # Training configuration
        batch_size = 1  # Keep batch size at 1
        grad_accum_steps = 32  # Increase gradient accumulation steps
        num_epochs = 50
        learning_rate = 1e-4
        checkpoint_frequency = 5
        
        # Early stopping configuration
        early_stopping_patience = 5
        early_stopping_min_delta = 1e-4
        best_loss = float('inf')
        patience_counter = 0
        
        # Initialize optimizer with gradient clipping and memory optimizations
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
            for chunk_idx in range(num_pseudobulk_chunks):
                # Calculate number of batches in this chunk
                chunk = np.load(f'data/chunks/pseudobulk/chunk_{chunk_idx}.npy')
                num_batches_in_chunk = (len(chunk) + batch_size - 1) // batch_size
                del chunk
                gc.collect()
                jax.clear_caches()
                
                # Process each batch in the chunk
                for batch_idx in range(num_batches_in_chunk):
                    # Load batch from pseudobulk chunk
                    pseudobulk_batch = process_batch_from_chunks(
                        chunk_idx, batch_idx, batch_size,
                        'data/chunks/pseudobulk'
                    )
                    
                    # Load corresponding batch from celltype chunk
                    celltype_batch = process_batch_from_chunks(
                        chunk_idx, batch_idx, batch_size,
                        'data/chunks/celltype'
                    )
                    
                    rng_key = jax.random.PRNGKey(epoch * 1000 + chunk_idx * 100 + batch_idx)
                    parameters, opt_state, batch_loss = train_step(
                        parameters,
                        opt_state,
                        pseudobulk_batch,
                        celltype_batch,
                        forward_fn,
                        optimizer,
                        rng_key,
                        grad_accum_steps
                    )
                    
                    epoch_loss += batch_loss
                    num_batches += 1
                    
                    # Clear memory after each batch
                    del pseudobulk_batch
                    del celltype_batch
                    gc.collect()
                    jax.clear_caches()
            
            # Compute average loss
            avg_loss = epoch_loss / num_batches
            
            # Compute embeddings for evaluation in chunks
            logging.info("Computing embeddings for evaluation...")
            os.makedirs(f'data/embeddings/epoch_{epoch}', exist_ok=True)
            
            # Process pseudobulk data in chunks
            for chunk_idx in range(num_pseudobulk_chunks):
                chunk = np.load(f'data/chunks/pseudobulk/chunk_{chunk_idx}.npy')
                num_batches_in_chunk = (len(chunk) + batch_size - 1) // batch_size
                
                # Process each batch in the chunk
                chunk_embeddings = []
                for batch_idx in range(num_batches_in_chunk):
                    batch = process_batch_from_chunks(
                        chunk_idx, batch_idx, batch_size,
                        'data/chunks/pseudobulk'
                    )
                    batch_embeddings = process_batch(batch, parameters, forward_fn, jax.random.PRNGKey(0))
                    chunk_embeddings.append(batch_embeddings)
                    del batch
                    del batch_embeddings
                    gc.collect()
                    jax.clear_caches()
                
                # Save chunk embeddings using memory-mapped array
                chunk_embeddings = np.vstack(chunk_embeddings)
                save_embeddings_memmap(
                    f'data/embeddings/epoch_{epoch}',
                    f'data/embeddings/epoch_{epoch}/pseudobulk_chunk_{chunk_idx}.npy',
                    chunk_embeddings.shape
                )
                del chunk
                del chunk_embeddings
                gc.collect()
                jax.clear_caches()
            
            # Process celltype data in chunks
            for chunk_idx in range(num_celltype_chunks):
                chunk = np.load(f'data/chunks/celltype/chunk_{chunk_idx}.npy')
                num_batches_in_chunk = (len(chunk) + batch_size - 1) // batch_size
                
                # Process each batch in the chunk
                chunk_embeddings = []
                for batch_idx in range(num_batches_in_chunk):
                    batch = process_batch_from_chunks(
                        chunk_idx, batch_idx, batch_size,
                        'data/chunks/celltype'
                    )
                    batch_embeddings = process_batch(batch, parameters, forward_fn, jax.random.PRNGKey(0))
                    chunk_embeddings.append(batch_embeddings)
                    del batch
                    del batch_embeddings
                    gc.collect()
                    jax.clear_caches()
                
                # Save chunk embeddings using memory-mapped array
                chunk_embeddings = np.vstack(chunk_embeddings)
                save_embeddings_memmap(
                    f'data/embeddings/epoch_{epoch}',
                    f'data/embeddings/epoch_{epoch}/celltype_chunk_{chunk_idx}.npy',
                    chunk_embeddings.shape
                )
                del chunk
                del chunk_embeddings
                gc.collect()
                jax.clear_caches()
            
            # Compute metrics in a streaming way
            metrics = compute_metrics_streaming(
                f'data/embeddings/epoch_{epoch}',
                num_pseudobulk_chunks,
                labels
            )
            
            # Early stopping check
            if avg_loss < (best_loss - early_stopping_min_delta):
                best_loss = avg_loss
                patience_counter = 0
                # Save best model with metrics
                save_checkpoint(
                    parameters, opt_state, epoch + 1, avg_loss, metrics,
                    "checkpoints/best_model.pkl"
                )
                logging.info(f"New best model saved with loss: {avg_loss:.4f}")
            else:
                patience_counter += 1
                logging.info(f"No improvement for {patience_counter} epochs")
            
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
                save_checkpoint(
                    parameters, opt_state, epoch + 1, avg_loss, metrics,
                    f"checkpoints/model_epoch_{epoch + 1}.pkl"
                )
        
        # Final evaluation
        logging.info("Performing final evaluation...")
        final_metrics = evaluate_model_streaming(
            f'data/embeddings/epoch_{epoch}',
            num_pseudobulk_chunks,
            labels,
            output_dir='final_evaluation'
        )
        
        # Get embedding dimension from first chunk
        first_chunk = np.load(glob.glob(f'data/embeddings/epoch_{epoch}/pseudobulk_chunk_*.npy')[0])
        embedding_dim = first_chunk.shape[1]
        del first_chunk
        gc.collect()
        
        # Save final embeddings using memory-mapped arrays
        logging.info("Saving final embeddings...")
        save_embeddings_memmap(
            f'data/embeddings/epoch_{epoch}',
            'data/trained_pseudobulk_embeddings.npy',
            (len(pseudobulk_indices), embedding_dim)
        )
        save_embeddings_memmap(
            f'data/embeddings/epoch_{epoch}',
            'data/trained_celltype_embeddings.npy',
            (len(celltype_indices), embedding_dim)
        )
        
        # Save as CSV with indices (in chunks)
        chunk_size = 1000
        for i in range(0, len(pseudobulk_indices), chunk_size):
            chunk_end = min(i + chunk_size, len(pseudobulk_indices))
            chunk_embeddings = np.load(f'data/embeddings/epoch_{epoch}/pseudobulk_chunk_{i//chunk_size}.npy')
            pd.DataFrame(
                chunk_embeddings,
                index=pseudobulk_indices[i:chunk_end]
            ).to_csv(f'data/trained_pseudobulk_embeddings_{i//chunk_size}.csv')
            del chunk_embeddings
            gc.collect()
        
        for i in range(0, len(celltype_indices), chunk_size):
            chunk_end = min(i + chunk_size, len(celltype_indices))
            chunk_embeddings = np.load(f'data/embeddings/epoch_{epoch}/celltype_chunk_{i//chunk_size}.npy')
            pd.DataFrame(
                chunk_embeddings,
                index=celltype_indices[i:chunk_end]
            ).to_csv(f'data/trained_celltype_embeddings_{i//chunk_size}.csv')
            del chunk_embeddings
            gc.collect()
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 