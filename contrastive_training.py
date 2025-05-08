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

# Configure JAX for memory efficiency
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.DEFAULT)
jax.config.update('jax_enable_x64', False)

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
        
        # Training parameters
        batch_size = 1
        num_epochs = 50
        learning_rate = 1e-4
        patience = 5  # Number of epochs to wait for improvement
        min_delta = 0.001  # Minimum change in metric to be considered as improvement
        
        # Initialize optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(parameters)
        
        # Initialize training history
        history = {
            'epoch': [],
            'loss': [],
            'mean_cosine_similarity': [],
            'silhouette_score': []
        }
        
        # Early stopping variables
        best_metric = float('-inf')
        best_epoch = 0
        best_params = None
        no_improvement_count = 0
        
        # Training loop
        logging.info("Starting training...")
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Create batches
            pseudobulk_batches = create_batches(pseudobulk_tokens, batch_size)
            celltype_batches = create_batches(celltype_tokens, batch_size)
            
            # Process batches
            for pseudobulk_batch, celltype_batch in tqdm(zip(pseudobulk_batches, celltype_batches), 
                                                       desc=f"Epoch {epoch + 1}/{num_epochs}"):
                rng_key = jax.random.PRNGKey(epoch)
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
                
                # Clear memory
                gc.collect()
                jax.clear_caches()
            
            # Compute average loss
            avg_loss = epoch_loss / num_batches
            
            # Compute embeddings for evaluation
            pseudobulk_embeddings = []
            celltype_embeddings = []
            
            for batch in create_batches(pseudobulk_tokens, batch_size):
                outs = forward_fn.apply(parameters, jax.random.PRNGKey(0), batch)
                batch_embeddings = np.array(outs["embeddings_4"].mean(axis=1))
                pseudobulk_embeddings.append(batch_embeddings)
            
            for batch in create_batches(celltype_tokens, batch_size):
                outs = forward_fn.apply(parameters, jax.random.PRNGKey(0), batch)
                batch_embeddings = np.array(outs["embeddings_4"].mean(axis=1))
                celltype_embeddings.append(batch_embeddings)
            
            pseudobulk_embeddings = np.vstack(pseudobulk_embeddings)
            celltype_embeddings = np.vstack(celltype_embeddings)
            
            # Compute metrics
            metrics = compute_similarity_metrics(pseudobulk_embeddings, celltype_embeddings, labels)
            current_metric = metrics['mean_cosine_similarity']
            
            # Update history
            history['epoch'].append(epoch + 1)
            history['loss'].append(avg_loss)
            history['mean_cosine_similarity'].append(current_metric)
            history['silhouette_score'].append(metrics['silhouette_score'])
            
            # Log epoch statistics
            logging.info(f"Epoch {epoch + 1}/{num_epochs}")
            logging.info(f"Average Loss: {avg_loss:.4f}")
            logging.info(f"Mean Cosine Similarity: {current_metric:.4f}")
            logging.info(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
            
            # Track training progress
            track_training_progress(history)
            
            # Early stopping check
            if current_metric > best_metric + min_delta:
                best_metric = current_metric
                best_epoch = epoch
                best_params = parameters
                no_improvement_count = 0
                
                # Save best model
                checkpoint_path = "checkpoints/best_model.pkl"
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(best_params, f)
                logging.info(f"New best model saved! Metric: {best_metric:.4f}")
            else:
                no_improvement_count += 1
                logging.info(f"No improvement for {no_improvement_count} epochs")
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f"checkpoints/model_epoch_{epoch + 1}.pkl"
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(parameters, f)
                logging.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Early stopping
            if no_improvement_count >= patience:
                logging.info(f"Early stopping triggered! No improvement for {patience} epochs")
                logging.info(f"Best model was from epoch {best_epoch + 1} with metric {best_metric:.4f}")
                break
        
        # Load best model for final evaluation
        if best_params is not None:
            parameters = best_params
            logging.info("Loading best model for final evaluation...")
        
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