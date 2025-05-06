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

def apply_chunk_attention(attention_fn, x, chunk_size, mask=None, rng_key=None):
    """Apply attention function with chunk-based processing.
    
    Args:
        attention_fn: Original attention function (already transformed)
        x: Input tensor of shape [batch_size, seq_len] (token IDs) or [batch_size, seq_len, hidden_dim] (embeddings)
        chunk_size: Size of each chunk
        mask: Optional additional mask
        rng_key: Random key for initialization
        
    Returns:
        Output tensor with chunk-based attention
    """
    # Log input shape
    logging.info(f"Input tensor shape in apply_chunk_attention: {x.shape}")
    
    # Handle both 2D (token IDs) and 3D (embeddings) inputs
    if len(x.shape) == 2:
        batch_size, seq_len = x.shape
        # forward_fn will handle the embedding internally
        logging.info("Processing token IDs (2D input)")
    elif len(x.shape) == 3:
        batch_size, seq_len, hidden_dim = x.shape
        logging.info("Processing pre-embedded vectors (3D input)")
    else:
        raise ValueError(f"Unexpected input shape: {x.shape}. Expected 2D (token IDs) or 3D (embeddings)")
    
    # Initialize output tensor
    if len(x.shape) == 2:
        # For token IDs, we'll let forward_fn handle the embedding
        output = jnp.zeros((batch_size, seq_len), dtype=x.dtype)
    else:
        # For embeddings, maintain the same shape
        output = jnp.zeros_like(x)
    
    # Initialize parameters once for the entire sequence
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    
    # Create a wrapper function that handles the chunk-based attention
    def chunk_attention_forward_fn(x):
        # Create attention mask for the current chunk
        chunk_size = x.shape[1]
        chunk_mask = jnp.ones((batch_size, chunk_size, chunk_size), dtype=bool)
        
        # Combine with additional mask if provided
        if mask is not None:
            chunk_mask = jnp.logical_and(chunk_mask, mask[:, :chunk_size, :chunk_size])
        
        # Apply attention with the mask
        return attention_fn.apply(attention_fn.init(rng_key, x), rng_key, x, mask=chunk_mask)
    
    # Transform the function with Haiku once
    chunk_attention_forward_fn = hk.transform(chunk_attention_forward_fn)
    
    # Initialize parameters once with a small chunk
    init_chunk_size = min(chunk_size, 128)  # Use smaller size for initialization
    if len(x.shape) == 2:
        params = chunk_attention_forward_fn.init(rng_key, x[:, :init_chunk_size])
    else:
        params = chunk_attention_forward_fn.init(rng_key, x[:, :init_chunk_size, :])
    
    # Process each chunk
    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        current_chunk_size = chunk_end - chunk_start
        
        # Extract current chunk
        if len(x.shape) == 2:
            chunk = x[:, chunk_start:chunk_end]
        else:
            chunk = x[:, chunk_start:chunk_end, :]
        
        # Process chunk using the same parameters
        chunk_output = chunk_attention_forward_fn.apply(params, rng_key, chunk)
        
        # Log shape after first chunk to verify embedding
        if chunk_start == 0:
            logging.info(f"Output shape after first chunk: {chunk_output.shape}")
        
        # Store chunk output in the corresponding position
        if len(x.shape) == 2:
            # For token IDs, the output will be embeddings
            output = output.at[:, chunk_start:chunk_end].set(chunk_output)
        else:
            # For embeddings, maintain the same shape
            output = output.at[:, chunk_start:chunk_end, :].set(chunk_output)
    
    return output

def process_batch(batch_tokens, parameters, forward_fn, random_key, chunk_size):
    """Process a single batch of tokens with chunk-based attention.
    
    Args:
        batch_tokens: Input tokens of shape [batch_size, seq_len]
        parameters: Model parameters
        forward_fn: Forward function (already transformed)
        random_key: Random key
        chunk_size: Size of each chunk for attention
        
    Returns:
        Processed embeddings of shape [batch_size, hidden_dim]
    """
    try:
        # Log input shape
        logging.info(f"Input batch shape in process_batch: {batch_tokens.shape}")
        
        # Create a wrapper function that handles the chunk-based attention
        def chunk_attention_forward_fn(x):
            # Apply the forward function with chunking
            return apply_chunk_attention(forward_fn, x, chunk_size, rng_key=random_key)
        
        # Transform the function with Haiku once
        chunk_attention_forward_fn = hk.transform(chunk_attention_forward_fn)
        
        # Initialize parameters once with a small chunk
        init_chunk_size = min(chunk_size, 128)  # Use smaller size for initialization
        chunk_params = chunk_attention_forward_fn.init(random_key, batch_tokens[:, :init_chunk_size])
        
        # Process in smaller sub-chunks to avoid memory issues
        batch_size = batch_tokens.shape[0]
        sub_chunk_size = min(32, batch_size)  # Process at most 32 samples at once
        all_embeddings = []
        
        for i in range(0, batch_size, sub_chunk_size):
            sub_chunk_end = min(i + sub_chunk_size, batch_size)
            sub_chunk = batch_tokens[i:sub_chunk_end]
            
            # Apply the function with the same parameters
            outs = chunk_attention_forward_fn.apply(
                {**parameters, **chunk_params},  # Combine both parameter sets
                random_key,
                sub_chunk
            )
            
            # Log shape after first sub-chunk
            if i == 0:
                logging.info(f"Output shape after first sub-chunk: {outs['embeddings_4'].shape}")
            
            # Get embeddings and mean pool
            sub_chunk_embeddings = np.array(outs["embeddings_4"].mean(axis=1), dtype=np.float32)
            all_embeddings.append(sub_chunk_embeddings)
            
            # Clear memory
            del outs
            del sub_chunk_embeddings
            gc.collect()
            jax.clear_caches()
        
        # Combine all sub-chunk embeddings
        batch_embeddings = np.vstack(all_embeddings)
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
        batch_size = 32  # Reduced to avoid memory issues
        attention_chunk_size = 128  # Reduced to match model's expectations
        processing_chunk_size = 128  # Match attention_chunk_size for consistency
        num_samples = len(rna_seq_df)
        
        all_embeddings = []
        
        for chunk_start in range(0, num_samples, processing_chunk_size):
            chunk_end = min(chunk_start + processing_chunk_size, num_samples)
            logging.info(f"Processing chunk {chunk_start}-{chunk_end} of {num_samples} samples...")
            
            try:
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
                    batch_end = min(i + batch_size, len(tokens))
                    batch_tokens = tokens[i:batch_end]
                    random_key = jax.random.PRNGKey(0)
                    
                    try:
                        batch_embeddings = process_batch(batch_tokens, parameters, forward_fn, random_key, attention_chunk_size)
                        chunk_embeddings[i:i + batch_size] = batch_embeddings
                        
                        # Clear memory after each batch
                        del batch_embeddings
                        gc.collect()
                        jax.clear_caches()
                        
                        if (i // batch_size) % 10 == 0:  # Log progress every 10 batches
                            logging.info(f"Processed batch {i//batch_size + 1}/{(len(tokens) + batch_size - 1)//batch_size}")
                            
                    except Exception as e:
                        logging.error(f"Error in batch {i}: {str(e)}")
                        raise
                
                all_embeddings.append(chunk_embeddings)
                
            except Exception as e:
                logging.error(f"Error processing chunk {chunk_start}-{chunk_end}: {str(e)}")
                raise
            finally:
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