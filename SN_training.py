import os
import torch
import pickle
import json
import argparse
import numpy as np
from siamese_network.siamese_network import SiameseNetwork

def train_and_eval(
        exp_descr,
        dataset_path,
        results_dir,
        batch_size,
        lr,
        contrastive_learning_num_epochs,
        contrastive_learning_iteration,
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

    siamese_net = SiameseNetwork(
        batch_size=batch_size,
        lr=lr,
        contrastive_learning_num_epochs=contrastive_learning_num_epochs,
        contrastive_learning_iteration=contrastive_learning_iteration,
        tau=tau,
        k=k,
        seed=seed,
        device='cuda',
        best_model=None
    )
    
    print("Constrastive learning")
    contrastive_history = siamese_net.fit(X_train=X_train, y_train=y_train,
                                                    X_val=X_val, y_val=y_val)
    
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
        'tau': tau, 'k': k, 'seed': seed
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