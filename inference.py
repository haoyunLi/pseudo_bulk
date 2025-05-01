import haiku as hk
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from multiomics_open_research.bulk_rna_bert.pretrained import get_pretrained_model
from multiomics_open_research.bulk_rna_bert.preprocess import preprocess_rna_seq_for_bulkrnabert
import gc

# Configure JAX for memory efficiency
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
jax.config.update('jax_enable_x64', False)

# Get pretrained model
print("Loading pretrained model...")
parameters, forward_fn, tokenizer, config = get_pretrained_model(
    model_name="bulk_rna_bert_tcga",
    embeddings_layers_to_save=(4,),
    checkpoint_directory="multiomics-open-research/checkpoints/",
)
forward_fn = hk.transform_with_state(forward_fn)

# Get bulk RNASeq data and tokenize it
print("Loading and preprocessing data...")
rna_seq_df = pd.read_csv("data/processed_pseudobulk_expression_W.csv", index_col=0)
# Convert all columns to numeric, coercing errors to NaN
rna_seq_df = rna_seq_df.apply(pd.to_numeric, errors='coerce')
# Fill any NaN values with 0
rna_seq_df = rna_seq_df.fillna(0)
rna_seq_array = preprocess_rna_seq_for_bulkrnabert(rna_seq_df, config)
tokens_ids = tokenizer.batch_tokenize(rna_seq_array)
tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)

# Process in very small batches to avoid memory issues
batch_size = 4  # Reduced batch size
num_samples = len(tokens)
all_embeddings = []

print(f"Processing {num_samples} samples in batches of {batch_size}...")

# First, get the embedding dimension by processing a single sample
test_batch = tokens[:1]
test_outs, _ = forward_fn.apply(parameters, None, jax.random.PRNGKey(0), test_batch)
embedding_dim = test_outs["embeddings_4"].mean(axis=1).shape[1]
print(f"Embedding dimension: {embedding_dim}")

# Pre-allocate memory for embeddings
all_embeddings = np.zeros((num_samples, embedding_dim), dtype=np.float32)

for i in range(0, num_samples, batch_size):
    batch_tokens = tokens[i:i + batch_size]
    
    # Inference for this batch
    random_key = jax.random.PRNGKey(0)
    outs, _ = forward_fn.apply(parameters, None, random_key, batch_tokens)
    
    # Get mean embeddings from layer 4 for this batch
    batch_embeddings = np.array(outs["embeddings_4"].mean(axis=1))
    all_embeddings[i:i + batch_size] = batch_embeddings
    
    # Clear memory
    del outs
    del batch_embeddings
    gc.collect()
    jax.clear_caches()
    
    if (i // batch_size) % 10 == 0:  # Print progress every 10 batches
        print(f"Processed batch {i//batch_size + 1}/{(num_samples + batch_size - 1)//batch_size}")

print("Saving results...")
# Save embeddings
np.save('data/mean_embeddings.npy', all_embeddings)

# Also save as CSV with donor IDs as index
mean_embedding_df = pd.DataFrame(
    all_embeddings,
    index=rna_seq_df.index
)
mean_embedding_df.to_csv('data/mean_embeddings.csv')

print("Done!")