import haiku as hk
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from multiomics_open_research.bulk_rna_bert.pretrained import get_pretrained_model
from multiomics_open_research.bulk_rna_bert.preprocess import preprocess_rna_seq_for_bulkrnabert

# Get pretrained model
parameters, forward_fn, tokenizer, config = get_pretrained_model(
    model_name="bulk_rna_bert_tcga",
    embeddings_layers_to_save=(4,),
    checkpoint_directory="multiomics-open-research/checkpoints/",
)
forward_fn = hk.transform(forward_fn)

# Get bulk RNASeq data and tokenize it
rna_seq_df = pd.read_csv("data/processed_pseudobulk_expression.csv", index_col=0)
# Convert all columns to numeric, coercing errors to NaN
rna_seq_df = rna_seq_df.apply(pd.to_numeric, errors='coerce')
# Fill any NaN values with 0
rna_seq_df = rna_seq_df.fillna(0)
rna_seq_array = preprocess_rna_seq_for_bulkrnabert(rna_seq_df, config)
tokens_ids = tokenizer.batch_tokenize(rna_seq_array)
tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)

# Inference
random_key = jax.random.PRNGKey(0)
outs = forward_fn.apply(parameters, random_key, tokens)

# Get mean embeddings from layer 4
mean_embedding = outs["embeddings_4"].mean(axis=1)

# Convert to numpy array and save to file
mean_embedding_np = np.array(mean_embedding)
np.save('data/mean_embeddings.npy', mean_embedding_np)

# Also save as CSV with donor IDs as index
mean_embedding_df = pd.DataFrame(
    mean_embedding_np,
    index=rna_seq_df.index
)
mean_embedding_df.to_csv('data/mean_embeddings.csv')