import haiku as hk
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from multiomics_open_research.bulk_rna_bert.pretrained import get_pretrained_model
from multiomics_open_research.bulk_rna_bert.preprocess import preprocess_rna_seq_for_bulkrnabert
import gc
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Optimize JAX configuration for GPU
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_disable_jit', False)  # Enable JIT compilation
jax.config.update('jax_threefry_partitionable', True)  # Enable better parallelization

def process_batch(batch_tokens, parameters, forward_fn, random_key):
    """Process a single batch of tokens.
    
    Args:
        batch_tokens: Input tokens of shape [batch_size, seq_len]
        parameters: Model parameters
        forward_fn: Forward function (already transformed)
        random_key: Random key
        
    Returns:
        Processed embeddings of shape [batch_size, hidden_dim]
    """
    try:
        # Log input shape
        logging.info(f"Input batch shape in process_batch: {batch_tokens.shape}")
        
        # Apply the forward function directly
        outs = forward_fn.apply(parameters, random_key, batch_tokens)
        
        # Log output shape
        logging.info(f"Output shape: {outs['embeddings_4'].shape}")
        
        # Get embeddings and mean pool
        batch_embeddings = np.array(outs["embeddings_4"].mean(axis=1), dtype=np.float32)
        logging.info(f"Final batch embeddings shape: {batch_embeddings.shape}")
        
        return batch_embeddings
        
    except Exception as e:
        logging.error(f"Error processing batch: {str(e)}")
        raise

def main():
    try:
        # Get pretrained model
        logging.info("Loading pretrained model...")
        parameters, forward_fn, tokenizer, config = get_pretrained_model(
            model_name="bulk_rna_bert_tcga",
            embeddings_layers_to_save=(4,),
            checkpoint_directory="multiomics-open-research/checkpoints/",
        )
        forward_fn = hk.transform(forward_fn)

        # Get bulk RNASeq data and tokenize it
        logging.info("Loading and preprocessing data...")
        rna_seq_df = pd.read_csv("data/celltype_specific_2d_matrix.csv", index_col=0)
        rna_seq_df = rna_seq_df.apply(pd.to_numeric, errors='coerce')
        rna_seq_df = rna_seq_df.fillna(0)
        
        # Configuration
        batch_size = 1 
        num_samples = len(rna_seq_df)
        
        all_embeddings = []
        
        # Process data in batches
        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            logging.info(f"Processing batch {i}-{batch_end} of {num_samples} samples...")
            
            try:
                # Process current batch
                batch_df = rna_seq_df.iloc[i:batch_end]
                rna_seq_array = preprocess_rna_seq_for_bulkrnabert(batch_df, config)
                tokens_ids = tokenizer.batch_tokenize(rna_seq_array)
                tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)
                
                # Get embedding dimension from first sample
                if i == 0:
                    test_batch = tokens[:1]
                    test_outs = forward_fn.apply(parameters, jax.random.PRNGKey(0), test_batch)
                    embedding_dim = test_outs["embeddings_4"].mean(axis=1).shape[1]
                    logging.info(f"Embedding dimension: {embedding_dim}")
                
                # Process batch
                random_key = jax.random.PRNGKey(0)
                batch_embeddings = process_batch(tokens, parameters, forward_fn, random_key)
                all_embeddings.append(batch_embeddings)
                
                # Clear memory after each batch
                del batch_df
                del rna_seq_array
                del tokens_ids
                del tokens
                del batch_embeddings
                gc.collect()
                jax.clear_caches()
                
            except Exception as e:
                logging.error(f"Error processing batch {i}-{batch_end}: {str(e)}")
                raise
        
        # Combine all embeddings
        all_embeddings = np.vstack(all_embeddings)
        
        logging.info("Saving results...")
        # Save embeddings
        np.save('data/celltype_specific_embeddings.npy', all_embeddings)
        
        # Also save as CSV with donor IDs as index
        mean_embedding_df = pd.DataFrame(
            all_embeddings,
            index=rna_seq_df.index
        )
        mean_embedding_df.to_csv('data/celltype_specific_embeddings.csv')
        
        logging.info("Done!")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()