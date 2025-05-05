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

def apply_chunk_attention(attention_fn, x, chunk_size, mask=None):
    """Apply attention function with chunk-based masking.
    
    Args:
        attention_fn: Original attention function
        x: Input tensor
        chunk_size: Size of each chunk
        mask: Optional additional mask
        
    Returns:
        Output tensor with chunk-based attention
    """
    seq_len = x.shape[1]
    chunk_mask = create_chunk_attention_mask(seq_len, chunk_size)
    
    # Combine with additional mask if provided
    if mask is not None:
        chunk_mask = jnp.logical_and(chunk_mask, mask)
    
    # Apply attention with chunk mask
    return attention_fn(x, mask=chunk_mask)

def process_batch(batch_tokens, parameters, forward_fn, random_key, chunk_size):
    """Process a single batch of tokens with chunk-based attention."""
    try:
        # Modify the forward function to use chunk-based attention
        def chunk_attention_forward_fn(x, mask=None):
            return apply_chunk_attention(forward_fn, x, chunk_size, mask)
        
        outs = chunk_attention_forward_fn.apply(parameters, random_key, batch_tokens)
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
        
        # Configuration
        batch_size = 1  # Process one sample at a time
        attention_chunk_size = 128  # Size of attention chunks
        processing_chunk_size = 500  # Size of data processing chunks
        num_samples = len(rna_seq_df)
        
        all_embeddings = []
        
        for chunk_start in range(0, num_samples, processing_chunk_size):
            chunk_end = min(chunk_start + processing_chunk_size, num_samples)
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
                    batch_embeddings = process_batch(batch_tokens, parameters, forward_fn, random_key, attention_chunk_size)
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