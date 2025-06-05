import os
import torch
import pickle
import json
import argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from .SN import SN

def plot_embeddings(emb_train, y_train, emb_val, y_val, emb_test, y_test, y_mapping, path, seed=42):
    all_embs = np.vstack([emb_train, emb_val, emb_test])
    all_labels = np.concatenate([y_train, y_val, y_test])
    all_domains = np.array(['train'] * len(emb_train) + ['val'] * len(emb_val) + ['test'] * len(emb_test))
    
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=seed)
    all_embs_2d = tsne.fit_transform(all_embs)

    red_train = all_embs_2d[all_domains == 'train']
    red_val = all_embs_2d[all_domains == 'val']
    red_test = all_embs_2d[all_domains == 'test']

    labels_train = all_labels[all_domains == 'train']
    labels_val = all_labels[all_domains == 'val']
    labels_test = all_labels[all_domains == 'test']

    _, axs = plt.subplots(1, 3, figsize=(18, 6))
    subsets = [('Train', red_train, labels_train),
            ('Validation', red_val, labels_val),
            ('Test', red_test, labels_test)]
    
    colors = sns.color_palette("hls", len(np.unique(all_labels)))

    for ax, (title, reduced, labels) in zip(axs, subsets):
        for label in np.unique(labels):
            idx = labels == label
            ax.scatter(reduced[idx, 0], reduced[idx, 1],
                    label=f'Class {y_mapping[label]}', alpha=0.6, c=[colors[label]])
        ax.set_title(f't-SNE of {title} Embeddings')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.grid(True)

    axs[2].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig(path)

def train_and_eval(
        exp_descr,
        dataset_path,
        results_dir,
        batch_size,
        lr,
        contrastive_learning_num_epochs,
        contrastive_learning_iteration,
        dropout,
        tau,
        k,
        seed):
    
    results_subdir = os.path.join(results_dir, exp_descr)
    if not os.path.exists(results_subdir):
        os.makedirs(results_subdir)

    data = np.load(dataset_path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    y_mapping = data['y_mapping']

    siamese_net = SN(
        batch_size=batch_size,
        lr=lr,
        contrastive_learning_num_epochs=contrastive_learning_num_epochs,
        contrastive_learning_iteration=contrastive_learning_iteration,
        dropout=dropout,
        tau=tau,
        k=k,
        seed=seed,
        device='cuda',
        best_model=None
    )

    print("Contrastive learning")
    contrastive_history = siamese_net.fit(X_train=X_train, y_train=y_train,
                                                    X_val=X_val, y_val=y_val)
    
    print("Plotting embeddings")
    emb_train = siamese_net.get_embeddings(X_train)
    emb_val = siamese_net.get_embeddings(X_val)
    emb_test = siamese_net.get_embeddings(X_test)
    plot_embeddings(
        emb_train=emb_train, y_train=y_train,
        emb_val=emb_val, y_val=y_val,
        emb_test=emb_test, y_test=y_test,
        y_mapping=y_mapping,
        path=os.path.join(results_subdir, f"embeddings_plot.png"), seed=seed
    )
    emb_train = siamese_net.get_embeddings(X_train, siamese_net.last_model)
    emb_val = siamese_net.get_embeddings(X_val, siamese_net.last_model)
    emb_test = siamese_net.get_embeddings(X_test, siamese_net.last_model)
    plot_embeddings(
        emb_train=emb_train, y_train=y_train,
        emb_val=emb_val, y_val=y_val,
        emb_test=emb_test, y_test=y_test,
        y_mapping=y_mapping,
        path=os.path.join(results_subdir, f"last_model_embeddings_plot.png"), seed=seed
    )

    print("Testing")
    y_pred = siamese_net.predict(X_ref=X_train, y_ref=y_train, X_test=X_test)
    
    print("Saving model, training history and predictions")
    torch.save(siamese_net.best_model, os.path.join(results_subdir, f"SN.pkl"))
    with open(os.path.join(results_subdir, f"SN_contrastive_history.pkl"), 'wb') as f:
        pickle.dump(contrastive_history, f)
    np.save(os.path.join(results_subdir, f"SN_predictions.npy"), y_pred)
    exp_params = {
        'dataset_path': dataset_path,
        'batch_size': batch_size, 'lr': lr,
        'contrastive_learning_num_epochs': contrastive_learning_num_epochs,
        'contrastive_learning_iteration': contrastive_learning_iteration,
        'dropout': dropout, 'tau': tau, 'k': k, 'seed': seed
    }
    with open(os.path.join(results_subdir, f"experiment_settings.json"), 'w') as f:
        json.dump(exp_params, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_descr', type=str, required=True,
        help='Descriptive name of the experiment for saving the results.'
        'A folder with this name will be created in the results directory.'
        )
    parser.add_argument('--dataset_path', type=str, required=True,
        help='Path to the dataset file.'
        )
    parser.add_argument('--results_dir', type=str, required=True,
        help='Directory where the results will be saved.'
    )
    parser.add_argument('--batch_size', type=int, default=64,
        help='Batch size for training.'
    )
    parser.add_argument('--lr', type=float, default=1e-4,
        help='Learning rate for the optimizer.'
    )
    parser.add_argument('--contrastive_learning_num_epochs', type=int, default=20,
        help='Number of epochs for contrastive learning.'
    )
    parser.add_argument('--contrastive_learning_iteration', type=int, default=500,
        help='Number of iterations for contrastive learning.'
    )
    parser.add_argument('--dropout', type=float, default=0.0,
        help='Dropout rate for the model.'
    )
    parser.add_argument('--tau', type=float, default=1,
        help='Temperature parameter for contrastive loss.'
    )
    parser.add_argument('--k', type=int, default=3,
        help='Number of nearest neighbors for classification.'
    )
    parser.add_argument('--seed', type=int, default=42,
        help='Random seed for reproducibility.'
    )

    args = parser.parse_args()

    train_and_eval(**vars(args))