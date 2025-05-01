import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import TSNE
import umap
import logging
from tqdm import tqdm
import os
from sklearn.preprocessing import LabelEncoder

def compute_similarity_metrics(pseudobulk_embeddings, celltype_embeddings, labels):
    """
    Compute various similarity metrics between embeddings.
    
    Args:
        pseudobulk_embeddings: Embeddings from pseudobulk data
        celltype_embeddings: Embeddings from celltype-specific data
        labels: Ground truth labels for samples
        
    Returns:
        Dictionary containing various similarity metrics
    """
    metrics = {}
    
    # Compute cosine similarity between corresponding pairs
    similarities = []
    for i in range(len(pseudobulk_embeddings)):
        similarity = np.dot(pseudobulk_embeddings[i], celltype_embeddings[i]) / (
            np.linalg.norm(pseudobulk_embeddings[i]) * np.linalg.norm(celltype_embeddings[i])
        )
        similarities.append(similarity)
    metrics['mean_cosine_similarity'] = np.mean(similarities)
    metrics['std_cosine_similarity'] = np.std(similarities)
    
    # Compute silhouette score for clustering quality
    combined_embeddings = np.vstack([pseudobulk_embeddings, celltype_embeddings])
    combined_labels = np.concatenate([labels, labels])
    metrics['silhouette_score'] = silhouette_score(combined_embeddings, combined_labels)
    
    return metrics

def visualize_embeddings(pseudobulk_embeddings, celltype_embeddings, labels, label_encoder, output_dir='visualizations'):
    """
    Create various visualizations of the embeddings.
    
    Args:
        pseudobulk_embeddings: Embeddings from pseudobulk data
        celltype_embeddings: Embeddings from celltype-specific data
        labels: Numeric labels for samples
        label_encoder: LabelEncoder used to convert labels
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine embeddings for visualization
    combined_embeddings = np.vstack([pseudobulk_embeddings, celltype_embeddings])
    data_types = ['pseudobulk'] * len(pseudobulk_embeddings) + ['celltype'] * len(celltype_embeddings)
    
    # Convert numeric labels back to original labels for visualization
    original_labels = label_encoder.inverse_transform(labels)
    combined_labels = np.concatenate([original_labels, original_labels])
    
    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(combined_embeddings)
    
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], 
                   hue=data_types, style=combined_labels, palette='Set2')
    plt.title('t-SNE Visualization by Data Type')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], 
                   hue=combined_labels, style=data_types, palette='Set2')
    plt.title('t-SNE Visualization by Cell Type')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tsne_visualization.png', bbox_inches='tight')
    plt.close()
    
    # UMAP visualization
    reducer = umap.UMAP(random_state=42)
    umap_results = reducer.fit_transform(combined_embeddings)
    
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=umap_results[:, 0], y=umap_results[:, 1], 
                   hue=data_types, style=combined_labels, palette='Set2')
    plt.title('UMAP Visualization by Data Type')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=umap_results[:, 0], y=umap_results[:, 1], 
                   hue=combined_labels, style=data_types, palette='Set2')
    plt.title('UMAP Visualization by Cell Type')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/umap_visualization.png', bbox_inches='tight')
    plt.close()
    
    # Similarity matrix visualization
    similarity_matrix = np.dot(pseudobulk_embeddings, celltype_embeddings.T)
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, cmap='viridis')
    plt.title('Similarity Matrix between Pseudobulk and Celltype Embeddings')
    plt.xlabel('Celltype Samples')
    plt.ylabel('Pseudobulk Samples')
    plt.savefig(f'{output_dir}/similarity_matrix.png', bbox_inches='tight')
    plt.close()
    
    # Distribution of similarities
    plt.figure(figsize=(10, 6))
    diagonal_similarities = np.diag(similarity_matrix)
    off_diagonal_similarities = similarity_matrix[~np.eye(similarity_matrix.shape[0], dtype=bool)]
    
    sns.histplot(diagonal_similarities, label='Corresponding Pairs', alpha=0.5)
    sns.histplot(off_diagonal_similarities, label='Non-corresponding Pairs', alpha=0.5)
    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(f'{output_dir}/similarity_distribution.png', bbox_inches='tight')
    plt.close()

def track_training_progress(history, output_dir='training_history'):
    """
    Track and visualize training progress.
    
    Args:
        history: Dictionary containing training history
        output_dir: Directory to save training history
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training loss
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(history['epoch'], history['loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Time')
    plt.legend()
    
    # Plot metrics
    plt.subplot(2, 1, 2)
    metrics = ['mean_cosine_similarity', 'silhouette_score']
    for metric in metrics:
        plt.plot(history['epoch'], history[metric], label=metric)
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Training Metrics over Time')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_progress.png', bbox_inches='tight')
    plt.close()
    
    # Save history to CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv(f'{output_dir}/training_history.csv', index=False)
    
    # Print final metrics
    logging.info("\nFinal Training Metrics:")
    for metric in metrics:
        logging.info(f"{metric}: {history[metric][-1]:.4f}")
    logging.info(f"Final Loss: {history['loss'][-1]:.4f}")

def evaluate_model(pseudobulk_embeddings, celltype_embeddings, labels, label_encoder, output_dir='evaluation'):
    """
    Comprehensive evaluation of the model.
    
    Args:
        pseudobulk_embeddings: Embeddings from pseudobulk data
        celltype_embeddings: Embeddings from celltype-specific data
        labels: Numeric labels for samples
        label_encoder: LabelEncoder used to convert labels
        output_dir: Directory to save evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute metrics
    metrics = compute_similarity_metrics(pseudobulk_embeddings, celltype_embeddings, labels)
    
    # Save metrics with descriptions
    with open(f'{output_dir}/metrics.txt', 'w') as f:
        f.write("Model Evaluation Metrics:\n\n")
        f.write("1. Mean Cosine Similarity: Measures the average similarity between corresponding pseudobulk and celltype embeddings\n")
        f.write(f"   Value: {metrics['mean_cosine_similarity']:.4f}\n\n")
        f.write("2. Standard Deviation of Cosine Similarity: Measures the spread of similarity scores\n")
        f.write(f"   Value: {metrics['std_cosine_similarity']:.4f}\n\n")
        f.write("3. Silhouette Score: Measures how well the embeddings are clustered by cell type\n")
        f.write(f"   Value: {metrics['silhouette_score']:.4f}\n")
    
    # Create visualizations
    visualize_embeddings(pseudobulk_embeddings, celltype_embeddings, labels, label_encoder, output_dir)
    
    return metrics 