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
    logging.info(f"Pseudobulk token shape: {pseudobulk_tokens.shape}")
    logging.info(f"Celltype token shape: {celltype_tokens.shape}")
    logging.info(f"Number of unique labels: {len(np.unique(labels))}")
    
    return pseudobulk_tokens, celltype_tokens, pseudobulk_df.index, celltype_df.index, labels, label_encoder

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
        return compute_contrastive_loss(
            pseudobulk_batch,
            celltype_batch,
            forward_fn,
            params,
            rng_key
        )
    
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
        batch_size = 1  # Reduced to 1 to handle memory constraints
        grad_accum_steps = 8  # Accumulate gradients over 8 steps to maintain effective batch size
        num_epochs = 50
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
            
            # Process data in batches
            for i in range(0, len(pseudobulk_tokens), batch_size):
                batch_end = min(i + batch_size, len(pseudobulk_tokens))
                pseudobulk_batch = pseudobulk_tokens[i:batch_end]
                celltype_batch = celltype_tokens[i:batch_end]
                
                rng_key = jax.random.PRNGKey(epoch * 1000 + i)
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
                if i % 5 == 0:  # More frequent cleanup
                    gc.collect()
                    jax.clear_caches()
            
            # Compute average loss
            avg_loss = epoch_loss / num_batches
            
            # Compute embeddings for evaluation
            logging.info("Computing embeddings for evaluation...")
            pseudobulk_embeddings = []
            celltype_embeddings = []
            
            # Process pseudobulk data in batches
            for i in range(0, len(pseudobulk_tokens), batch_size):
                batch_end = min(i + batch_size, len(pseudobulk_tokens))
                batch = pseudobulk_tokens[i:batch_end]
                batch_embeddings = process_batch(batch, parameters, forward_fn, jax.random.PRNGKey(0))
                pseudobulk_embeddings.append(batch_embeddings)
                del batch
                del batch_embeddings
                gc.collect()
            
            # Process celltype data in batches
            for i in range(0, len(celltype_tokens), batch_size):
                batch_end = min(i + batch_size, len(celltype_tokens))
                batch = celltype_tokens[i:batch_end]
                batch_embeddings = process_batch(batch, parameters, forward_fn, jax.random.PRNGKey(0))
                celltype_embeddings.append(batch_embeddings)
                del batch
                del batch_embeddings
                gc.collect()
            
            # Combine embeddings
            pseudobulk_embeddings = np.vstack(pseudobulk_embeddings)
            celltype_embeddings = np.vstack(celltype_embeddings)
            
            # Compute metrics
            metrics = compute_similarity_metrics(pseudobulk_embeddings, celltype_embeddings, labels)
            
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