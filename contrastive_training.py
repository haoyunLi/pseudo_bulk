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

# Add memory optimization settings
jax.config.update('jax_gpu_memory_fraction', 0.9)  # Use 90% of GPU memory
jax.config.update('jax_gpu_memory_allocator', 'cuda_malloc_async')  # Use async memory allocator

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

def create_batches(tokens, batch_size):
    """Create batches from tokens."""
    num_samples = len(tokens)
    for i in range(0, num_samples, batch_size):
        yield tokens[i:i + batch_size]

def train_step(params, opt_state, pseudobulk_batch, celltype_batch, forward_fn, optimizer, rng_key):
    """Perform a single training step."""
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
        
        # Training parameters optimized for A100
        batch_size = 32  # Reduced from 128 to match inference settings
        num_epochs = 50  # Keep same number of epochs
        learning_rate = 1e-4
        checkpoint_frequency = 5  # Save checkpoint every 5 epochs
        
        # Initialize optimizer with gradient clipping
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Add gradient clipping
            optax.adam(learning_rate)
        )
        opt_state = optimizer.init(parameters)
        
        # Initialize training history
        history = {
            'epoch': [],
            'loss': [],
            'mean_cosine_similarity': [],
            'silhouette_score': []
        }
        
        # Training loop
        logging.info("Starting training...")
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Create batches
            pseudobulk_batches = create_batches(pseudobulk_tokens, batch_size)
            celltype_batches = create_batches(celltype_tokens, batch_size)
            
            # Process batches with progress tracking
            for batch_idx, (pseudobulk_batch, celltype_batch) in enumerate(tqdm(zip(pseudobulk_batches, celltype_batches), 
                                                       desc=f"Epoch {epoch + 1}/{num_epochs}")):
                rng_key = jax.random.PRNGKey(epoch * 1000 + batch_idx)  # Better RNG key generation
                parameters, opt_state, batch_loss = train_step(
                    parameters,
                    opt_state,
                    pseudobulk_batch,
                    celltype_batch,
                    forward_fn,
                    optimizer,
                    rng_key
                )
                
                epoch_loss += batch_loss
                num_batches += 1
                
                # Optimized memory clearing
                if batch_idx % 10 == 0:  # Clear memory every 10 batches
                    gc.collect()
                    jax.clear_caches()
            
            # Compute average loss
            avg_loss = epoch_loss / num_batches
            
            # Compute embeddings for evaluation in larger batches
            eval_batch_size = 64  # Reduced from 256 to match inference settings
            pseudobulk_embeddings = []
            celltype_embeddings = []
            
            for batch in create_batches(pseudobulk_tokens, eval_batch_size):
                outs = forward_fn.apply(parameters, jax.random.PRNGKey(0), batch)
                batch_embeddings = np.array(outs["embeddings_4"].mean(axis=1))
                pseudobulk_embeddings.append(batch_embeddings)
                del outs  # Clear memory immediately
                gc.collect()
            
            for batch in create_batches(celltype_tokens, eval_batch_size):
                outs = forward_fn.apply(parameters, jax.random.PRNGKey(0), batch)
                batch_embeddings = np.array(outs["embeddings_4"].mean(axis=1))
                celltype_embeddings.append(batch_embeddings)
                del outs  # Clear memory immediately
                gc.collect()
            
            pseudobulk_embeddings = np.vstack(pseudobulk_embeddings)
            celltype_embeddings = np.vstack(celltype_embeddings)
            
            # Compute metrics
            metrics = compute_similarity_metrics(pseudobulk_embeddings, celltype_embeddings, labels)
            
            # Update history
            history['epoch'].append(epoch + 1)
            history['loss'].append(avg_loss)
            history['mean_cosine_similarity'].append(metrics['mean_cosine_similarity'])
            history['silhouette_score'].append(metrics['silhouette_score'])
            
            # Log epoch statistics
            logging.info(f"Epoch {epoch + 1}/{num_epochs}")
            logging.info(f"Average Loss: {avg_loss:.4f}")
            logging.info(f"Mean Cosine Similarity: {metrics['mean_cosine_similarity']:.4f}")
            logging.info(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
            
            # Track training progress
            track_training_progress(history)
            
            # Save model checkpoint more frequently
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