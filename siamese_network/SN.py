import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import time
import copy

from .dataloader import BalancedBatchSampler, Dataset
from .model import CNN, ContrastiveLoss

def reproductibility(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm

class SN():
    def __init__(
            self,
            batch_size=64,
            lr=1e-4,
            contrastive_learning_num_epochs=20,
            contrastive_learning_iteration=500,
            dropout=0.0,
            tau=1,
            k=3,
            seed=42,
            device='cuda',
            best_model=None
        ) -> None:
        
        self.batch_size = batch_size
        self.lr = lr
        self.contrastive_learning_num_epochs = contrastive_learning_num_epochs
        self.contrastive_learning_iteration = contrastive_learning_iteration
        self.dropout = dropout
        self.tau = tau
        self.k = k
        self.seed = seed
        self.device = device
        self.best_model = best_model

    def fit(
            self,
            X_train,
            y_train,
            X_val=None,
            y_val=None,
            positive_ratio=1,
            negative_ratio=2
        ):
        
        reproductibility(self.seed)
        
        ### data loader ###
        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.1, random_state=self.seed)
    
        train_dataset = Dataset(X_train, y_train)
        train_batch_sampler = BalancedBatchSampler(
            y_train, positive_ratio=positive_ratio, negative_ratio=negative_ratio,
            iteration=self.contrastive_learning_iteration, batch_size=self.batch_size)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_sampler=train_batch_sampler, num_workers=4)
        
        ### define model & optimizer & loss function ###
        model = CNN(in_channels=X_train.shape[-1], dropout=self.dropout).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        loss_function = ContrastiveLoss(tau=self.tau)
        
        history = {
            "train_loss": [],
            "val_acc": []
        }
        
        ### Training ###
        best_acc = 0
        for epoch in range(self.contrastive_learning_num_epochs):
            start_time = time.time()
            ####################################
            model.train()
            train_losses = []
            for X1, X2, _, _, sim in tqdm(train_loader, leave=False):
                X1, X2, sim = X1.to(self.device), X2.to(self.device), sim.to(self.device)

                E1 = model(X1)
                E2 = model(X2)
                loss = loss_function(E1, E2, sim)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            ####################################
            model.eval()
            val_preds = self.predict(X_ref=X_train, y_ref=y_train, X_test=X_val, model=model)
            val_acc = accuracy_score(y_val, val_preds)
            ####################################
            end_time = time.time()
            epoch_time = end_time - start_time
            train_loss = torch.mean(torch.FloatTensor(train_losses)).item()

            if val_acc > best_acc:
                self.best_model = copy.deepcopy(model)
                best_acc = val_acc

            history["train_loss"].append(train_loss)
            history["val_acc"].append(val_acc)
            
            print('[%d/%d]-ptime: %.2f, train loss: %.6f, val acc: %.4f'
                    % ((epoch + 1), self.contrastive_learning_num_epochs, epoch_time, train_loss, val_acc))  
            
        self.last_model = model

        return history
    
    def get_embeddings(self, X, model=None):
        if model is None:
            model = self.best_model
        model.eval()

        loader = torch.utils.data.DataLoader(torch.tensor(X).float(), batch_size=self.batch_size)
        embs = []
        with torch.no_grad():
            for x in loader:
                x = x.to(self.device)
                emb = model(x)
                emb = F.normalize(emb, p=2, dim=1)
                embs.append(emb.cpu())
        return torch.cat(embs).numpy()
    
    def predict(self, X_ref, y_ref, X_test, model=None):
        emb_ref = self.get_embeddings(X_ref, model)
        emb_test = self.get_embeddings(X_test, model)

        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(emb_ref, y_ref)
        return knn.predict(emb_test)
