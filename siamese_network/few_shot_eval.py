import os
import json
import torch
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from .SN import SN

def plot_embeddings(emb_base, y_base, emb_new, y_new, y_mapping, path, seed=42):
    all_embs = np.vstack([emb_base, emb_new])
    all_labels = np.concatenate([y_base, y_new])

    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=seed)
    reduced_embs = tsne.fit_transform(all_embs)

    red_base = reduced_embs[:len(emb_base)]
    red_new = reduced_embs[len(emb_base):]

    unique_labels = np.unique(all_labels)
    colors = sns.color_palette("hls", len(unique_labels))

    plt.figure(figsize=(8, 6))
    
    for i, label in enumerate(unique_labels):
        color = colors[i]
        base_idx = y_base == label
        if np.any(base_idx):
            plt.scatter(red_base[base_idx, 0], red_base[base_idx, 1],
                        label=f'Class {y_mapping[label]}', alpha=0.6, color=color)
        new_idx = y_new == label
        if np.any(new_idx):
            plt.scatter(red_new[new_idx, 0], red_new[new_idx, 1],
                        label=f'Class {y_mapping[label]} (new)', alpha=0.6, color=color, marker='x')

    plt.title('t-SNE of embeddings')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig(path)

def few_shot_eval(
        exp_descr,
        base_dataset_path,
        novel_dataset_path,
        results_dir,
        model_checkpoint_path,
        model_settings_path,
        reference_dataset_size,
        seed
    ):

    results_subdir = os.path.join(results_dir, exp_descr)
    if not os.path.exists(results_subdir):
        os.makedirs(results_subdir)

    base_data = np.load(base_dataset_path)
    X_train = base_data['X_train']
    y_train = base_data['y_train']
    X_test = base_data['X_test']
    y_mapping = base_data['y_mapping']

    novel_data = np.load(novel_dataset_path)
    X_support = novel_data['X_support']
    y_support = novel_data['y_support']
    X_query = novel_data['X_query']
    y_query = novel_data['y_query']

    with open(model_settings_path, 'r') as f:
        model_settings = json.load(f)

    model = torch.load(model_checkpoint_path, weights_only=False)
    siamese_net = SN(
        batch_size=model_settings["batch_size"],
        lr=model_settings["lr"],
        contrastive_learning_num_epochs=model_settings["contrastive_learning_num_epochs"],
        contrastive_learning_iteration=model_settings["contrastive_learning_iteration"],
        dropout=model_settings["dropout"],
        tau=model_settings["tau"],
        k=model_settings["k"],
        seed=model_settings["seed"],
        device='cuda',
        best_model=model
    )

    X_train_support = np.concatenate((X_train, X_support), axis=0)
    y_train_support = np.concatenate((y_train, y_support), axis=0)
    y_pred_query_full = siamese_net.predict(X_ref=X_train_support, y_ref=y_train_support, X_test=X_query)
    np.save(os.path.join(results_subdir, f"SN_predictions_few_shot_full_ref.npy"), y_pred_query_full)

    X_train_sub, _, y_train_sub, _ = train_test_split(
        X_train, y_train, 
        train_size=reference_dataset_size, 
        stratify=y_train,
        random_state=seed
    )
    X_train_sub_support = np.concatenate((X_train_sub, X_support), axis=0)
    y_train_sub_support = np.concatenate((y_train_sub, y_support), axis=0)
    y_pred_query_sub = siamese_net.predict(X_ref=X_train_sub_support, y_ref=y_train_sub_support, X_test=X_query)
    np.save(os.path.join(results_subdir, f"SN_predictions_few_shot_sub_ref.npy"), y_pred_query_sub)

    y_pred_query_new = siamese_net.predict(X_ref=X_support, y_ref=y_support, X_test=X_query)
    np.save(os.path.join(results_subdir, f"SN_predictions_few_shot_new_ref.npy"), y_pred_query_new)

    X_new = np.concatenate((X_support, X_query), axis=0)
    y_new = np.concatenate((y_support, y_query), axis=0)
    emb_train = siamese_net.get_embeddings(X_train)
    emb_new = siamese_net.get_embeddings(X_new)
    plot_embeddings(
        emb_train, y_train,
        emb_new, y_new,
        y_mapping,
        path=os.path.join(results_subdir, f"embeddings_plot.png"),
        seed=seed
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_descr', type=str, required=True,
        help='Descriptive name of the experiment for saving the results.'
        'A folder with this name will be created in the results directory.'
    )
    parser.add_argument('--base_dataset_path', type=str, required=True,
        help='Path to the base dataset file.'
    )
    parser.add_argument('--novel_dataset_path', type=str, required=True,
        help='Path to the novel dataset file.'
    )
    parser.add_argument('--results_dir', type=str, required=True,
        help='Directory where the results will be saved.'
    )
    parser.add_argument('--model_checkpoint_path', type=str, required=True,
        help='Path to the model checkpoint file.'
    )
    parser.add_argument('--model_settings_path', type=str, required=True,
        help='Path to the model settings JSON file.'
    )
    parser.add_argument('--reference_dataset_size', type=int, required=True,
        help='Size of the reference dataset for few-shot evaluation.'
    )
    parser.add_argument('--seed', type=int, default=42,
        help='Random seed for reproducibility.'
    )
    
    args = parser.parse_args()

    few_shot_eval(**vars(args))