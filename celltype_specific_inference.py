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
    """Process a single batch of tokens."""
    try:
        outs = forward_fn.apply(parameters, random_key, batch_tokens)
        batch_embeddings = np.array(outs["embeddings_4"].mean(axis=1), dtype=np.float32)
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
        
        # Reduced batch size to prevent OOM
        batch_size = 32  # Reduced from 64
        num_samples = len(rna_seq_df)
        
        # Reduced chunk size for better memory management
        chunk_size = 2000  # Reduced from 5000
        all_embeddings = []
        
        for chunk_start in range(0, num_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_samples)
            logging.info(f"Processing chunk {chunk_start}-{chunk_end} of {num_samples} samples...")
            
            # Process current chunk
            chunk_df = rna_seq_df.iloc[chunk_start:chunk_end]
            rna_seq_array = preprocess_rna_seq_for_bulkrnabert(chunk_df, config)
            tokens_ids = tokenizer.batch_tokenize(rna_seq_array)
            tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)
            
            # Get embedding dimension from first sample
            if chunk_start == 0:
                test_batch = tokens[:1]
                test_outs = forward_fn.apply(parameters, jax.random.PRNGKey(0), test_batch)
                embedding_dim = test_outs["embeddings_4"].mean(axis=1).shape[1]
                logging.info(f"Embedding dimension: {embedding_dim}")
            
            # Process current chunk in batches
            chunk_embeddings = np.zeros((len(chunk_df), embedding_dim), dtype=np.float32)
            
            for i in range(0, len(tokens), batch_size):
                batch_tokens = tokens[i:i + batch_size]
                random_key = jax.random.PRNGKey(0)
                
                try:
                    batch_embeddings = process_batch(batch_tokens, parameters, forward_fn, random_key)
                    chunk_embeddings[i:i + batch_size] = batch_embeddings
                    
                    # Reduced frequency of memory clearing
                    if (i // batch_size) % 50 == 0:  # Only clear every 50 batches
                        gc.collect()
                        jax.clear_caches()
                        logging.info(f"Processed batch {i//batch_size + 1}/{(len(tokens) + batch_size - 1)//batch_size}")
                        
                except Exception as e:
                    logging.error(f"Error in batch {i}: {str(e)}")
                    raise
            
            all_embeddings.append(chunk_embeddings)
            
            # Clear memory after each chunk
            del chunk_df
            del rna_seq_array
            del tokens_ids
            del tokens
            del chunk_embeddings
            gc.collect()
            jax.clear_caches()
        
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