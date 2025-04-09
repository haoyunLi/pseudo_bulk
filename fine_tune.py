import haiku as hk
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from multiomics_open_research.bulk_rna_bert.pretrained import get_pretrained_model
from multiomics_open_research.bulk_rna_bert.preprocess import preprocess_rna_seq_for_bulkrnabert
import optax

def create_fine_tune_model(config):
    """Create a fine-tuned model with a new classification head."""
    def forward_fn(x, is_training=True):
        # Use the pretrained model as a feature extractor
        bert_output = hk.nets.BulkRNABERT(config)(x, is_training=is_training)
        
        # Get the mean embedding from layer 4 (as in the original model)
        mean_embedding = bert_output["embeddings_4"].mean(axis=1)
        
        # Add a new classification head
        # Adjust the output size based on task 
        num_classes = 2  # Change this based on task
        logits = hk.Linear(num_classes)(mean_embedding)
        
        return {"logits": logits, "embeddings": mean_embedding}
    
    return forward_fn

def main():
    # Load pretrained model
    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name="bulk_rna_bert_tcga",
        embeddings_layers_to_save=(4,),
        checkpoint_directory="multiomics-open-research/checkpoints/",
    )
    
    # Create fine-tuned model
    fine_tune_forward_fn = hk.transform(create_fine_tune_model(config))
    
    # Load and preprocess data
    rna_seq_df = pd.read_csv("data/processed_pseudobulk_expression.csv", index_col=0)
    rna_seq_df = rna_seq_df.apply(pd.to_numeric, errors='coerce')
    rna_seq_df = rna_seq_df.fillna(0)
    rna_seq_array = preprocess_rna_seq_for_bulkrnabert(rna_seq_df, config)
    tokens_ids = tokenizer.batch_tokenize(rna_seq_array)
    tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)
    
    # Initialize optimizer
    optimizer = optax.adam(learning_rate=1e-4)
    opt_state = optimizer.init(parameters)
    
    # Training loop
    num_epochs = 10
    batch_size = 32
    
    for epoch in range(num_epochs):
        # Shuffle data
        indices = np.random.permutation(len(tokens))
        
        for i in range(0, len(tokens), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_tokens = tokens[batch_indices]
            
            # batch_labels = ...
            
            def loss_fn(params):
                outputs = fine_tune_forward_fn.apply(params, None, batch_tokens)
               
                # loss = ...
                return loss
            
            # Calculate gradients and update parameters
            grads = jax.grad(loss_fn)(parameters)
            updates, opt_state = optimizer.update(grads, opt_state)
            parameters = optax.apply_updates(parameters, updates)
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            np.save(f'checkpoints/fine_tuned_model_epoch_{epoch+1}.npy', parameters)
    
    # Save final model
    np.save('checkpoints/fine_tuned_model_final.npy', parameters)

if __name__ == "__main__":
    main() 