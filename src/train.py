import json
import os
import shutil
import random
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

def seed_reproducibility(seed=42):
    """
    Fix random seeds for reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EmbeddingsDataset(Dataset):
    """
    Simple PyTorch dataset that yields (embedding, label).
    """
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        super().__init__()
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.embeddings[idx])  # float tensor
        y = torch.tensor(self.labels[idx])          # long tensor
        return x, y

class MLP(nn.Module):
    """
    A simple feed-forward network:
      fc1 -> ReLU -> Dropout -> fc2
    """
    def __init__(self, input_dim=192, hidden_dim=192, output_dim=2, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class FNNTrainer:
    """
    Trains a simple feed-forward neural network (MLP) on speaker embeddings.
    Saves logs to TensorBoard in runs/<model_name>-YYYY-MM-DD_HH-MM-SS,
    and copies misclassified audio to post-analysis/<model_name> folder.
    """

    def __init__(
        self,
        jsonl_path: str,
        model_name: str,
        embedding_dim: int = 192,
        hidden_dim: int = 192,
        batch_size: int = 2048,
        lr: float = 1e-3,
        dropout: float = 0.5,
        patience: int = 10,
        max_epochs: int = 10000,
        seed: int = 42
    ):
        """
        Args:
            jsonl_path : Path to embeddings JSONL file (must contain "embedding", "label", and "filepath")
            model_name : String identifier for logging and folder naming (e.g. 'ecapa' or 'resemblyzer')
            embedding_dim : Input dimension for MLP
            hidden_dim : Hidden dimension for MLP
            batch_size : Batch size for training
            lr : Learning rate
            dropout : Dropout probability in MLP
            patience : Early stopping patience
            max_epochs : Maximum number of epochs to train
            seed : Random seed
        """
        self.jsonl_path = jsonl_path
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.patience = patience
        self.max_epochs = max_epochs
        self.seed = seed

        # Make sure seeds are fixed for reproducibility.
        seed_reproducibility(self.seed)

        # Choose device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Internals for storing dataset and model
        self.dataset = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.model = None
        self.records = []  # We'll keep the original JSON records for post-analysis

        # Create post-analysis dir name under "post-analysis/model_name"
        self.post_analysis_dir = os.path.join("post-analysis", self.model_name)
        os.makedirs(self.post_analysis_dir, exist_ok=True)

        # Build TensorBoard run dir, e.g. runs/ecapa-2025-03-22_14-55-08
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join("runs", f"{self.model_name}-{current_time}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # This will be the path where we save the best model
        self.model_save_path = os.path.join(self.log_dir, "best_model.pth")

    def load_data(self):
        """
        Loads data from self.jsonl_path into memory, 
        builds PyTorch dataset and split into train/val/test.
        """
        with open(self.jsonl_path, "r") as f_in:
            for line in f_in:
                record = json.loads(line.strip())
                self.records.append(record)

        embeddings = []
        labels = []
        for r in self.records:
            emb = r["embedding"]
            lab = 0 if r["label"] == "M" else 1
            embeddings.append(emb)
            labels.append(lab)
        emb_array = np.array(embeddings, dtype=np.float32)
        label_array = np.array(labels, dtype=np.int64)

        self.dataset = EmbeddingsDataset(emb_array, label_array)

        # Train/val/test = 80/10/10
        num_total = len(self.dataset)
        train_size = int(0.8 * num_total)
        val_size = int(0.1 * num_total)
        test_size = num_total - train_size - val_size

        self.train_ds, self.val_ds, self.test_ds = random_split(
            self.dataset,
            [train_size, val_size, test_size]
        )

        self.train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_loader   = DataLoader(self.val_ds,   batch_size=self.batch_size, shuffle=False)
        self.test_loader  = DataLoader(self.test_ds,  batch_size=self.batch_size, shuffle=False)

    def build_model(self):
        """
        Creates the MLP model with user-specified dimensions and dropout.
        """
        self.model = MLP(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=2,
            dropout=self.dropout
        )
        self.model.to(self.device)

    def train_epoch(self, loader, criterion, optimizer):
        """
        Single epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()

            logits = self.model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def evaluate_epoch(self, loader, criterion):
        """
        Single epoch of evaluation (val or test).
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = criterion(logits, y)
                running_loss += loss.item() * x.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def train(self):
        """
        Main training loop with:
          - Load data, build model, set up optimizer & scheduler
          - Early stopping
          - TensorBoard logging
          - Save best model
        """
        # 1) Load the data if not already
        if self.dataset is None:
            self.load_data()

        # 2) Build the model if not built
        if self.model is None:
            self.build_model()

        # 3) Set up criterion, optimizer, scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # 4) Set up SummaryWriter
        writer = SummaryWriter(log_dir=self.log_dir)

        # Log model architecture
        writer.add_text("Model Architecture", str(self.model))

        # Log hyperparameters
        hyperparams = {
            "jsonl_path": self.jsonl_path,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "batch_size": self.batch_size,
            "learning_rate": self.lr,
            "optimizer": type(optimizer).__name__,
            "model_arch": type(self.model).__name__,
            "scheduler": "ReduceLROnPlateau",
            "dataset_size": len(self.dataset),
            "train_size": len(self.train_ds),
            "val_size": len(self.val_ds),
            "test_size": len(self.test_ds),
            "patience": self.patience,
            "max_epochs": self.max_epochs,
        }
        # Convert dict to nice string
        hyperparam_str = "\n".join(f"{k}: {v}" for k, v in hyperparams.items())
        writer.add_text("Hyperparameters", hyperparam_str)

        # 5) Training
        best_val_loss = float("inf")
        early_stop_counter = 0
        epochs_ran = 0

        for epoch in range(self.max_epochs):
            train_loss, train_acc = self.train_epoch(self.train_loader, criterion, optimizer)
            val_loss, val_acc = self.evaluate_epoch(self.val_loader, criterion)

            scheduler.step(val_loss)  # for ReduceLROnPlateau

            print(
                f"Epoch [{epoch+1}/{self.max_epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            # TensorBoard logging
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Acc/Train", train_acc, epoch)
            writer.add_scalar("Loss/Val", val_loss, epoch)
            writer.add_scalar("Acc/Val", val_acc, epoch)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("Learning_Rate", current_lr, epoch)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                torch.save(self.model.state_dict(), self.model_save_path)
                writer.add_text(
                    "Best Model",
                    f"Epoch {epoch+1} - Val Loss: {val_loss:.6f} - Val Acc: {val_acc:.6f}"
                )
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    writer.add_text(
                        "Training Info",
                        f"Early stopping triggered at epoch {epoch+1}"
                    )
                    epochs_ran = epoch+1
                    break
            epochs_ran = epoch+1

        # 6) Load best model, test set performance
        self.model.load_state_dict(torch.load(self.model_save_path))
        test_loss, test_acc = self.evaluate_epoch(self.test_loader, criterion)
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        writer.add_scalar("Loss/Test", test_loss, epochs_ran)
        writer.add_scalar("Acc/Test", test_acc, epochs_ran)
        writer.add_text("Test Results", f"Test Loss: {test_loss:.6f} - Test Acc: {test_acc:.6f}")

        # Final info
        final_info = {
            "total_epochs": epochs_ran,
            "best_val_loss": best_val_loss,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "early_stopped": early_stop_counter >= self.patience
        }
        final_info_str = "\n".join(f"{k}: {v}" for k, v in final_info.items())
        writer.add_text("Final Results", final_info_str)

        writer.close()
        print(f"Training complete. Logs and model saved to: {self.log_dir}")

    def post_analysis(self):
        """
        After training, gather misclassified audio files from train/val/test,
        copying them into subfolders inside post-analysis/<model_name>:
            post-analysis/<model_name>/{train|val|test}/{male|female}/
        Also prints accuracy on each subset.
        """
        # Make sure model is loaded
        if self.model is None:
            raise RuntimeError("Please call .train() first or build/load the model before post-analysis.")

        self.model.eval()

        # Helper for evaluating a single subset
        def analyze_and_save_errors(loader, subset, subset_name):
            """
            1) Runs the model on the provided DataLoader.
            2) Copies all misclassified audio files into:
                 post-analysis/<model_name>/<subset_name>/<male|female>/
            3) Prints the accuracy for that subset.
            """
            subset_dir = os.path.join(self.post_analysis_dir, subset_name)
            os.makedirs(subset_dir, exist_ok=True)

            correct_samples = 0
            total_samples = 0
            # subset.indices is how random_split tracks the original index in self.dataset

            with torch.no_grad():
                for batch_idx, (x_batch, y_batch) in enumerate(loader):
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    logits = self.model(x_batch)
                    preds = torch.argmax(logits, dim=1)

                    for i in range(len(x_batch)):
                        pred_label = preds[i].item()
                        true_label = y_batch[i].item()
                        total_samples += 1

                        if pred_label == true_label:
                            correct_samples += 1
                        else:
                            # misclassified
                            original_index = subset.indices[batch_idx * loader.batch_size + i]
                            audio_path = self.records[original_index]["filepath"]

                            subfolder_name = "male" if true_label == 0 else "female"
                            subfolder_path = os.path.join(subset_dir, subfolder_name)
                            os.makedirs(subfolder_path, exist_ok=True)

                            if os.path.isfile(audio_path):
                                file_name = os.path.basename(audio_path)
                                dest_path = os.path.join(subfolder_path, file_name)
                                shutil.copy2(audio_path, dest_path)

            accuracy = correct_samples / total_samples if total_samples > 0 else 0.0
            print(f"{subset_name} Accuracy: {accuracy * 100:.2f}% ({correct_samples} / {total_samples})")

        print("*" * 50)
        # Now analyze train, val, test
        print("Running post-analysis on Train set ...")
        analyze_and_save_errors(self.train_loader, self.train_ds, "train")

        print("\nRunning post-analysis on Validation set ...")
        analyze_and_save_errors(self.val_loader, self.val_ds, "val")

        print("\nRunning post-analysis on Test set ...")
        analyze_and_save_errors(self.test_loader, self.test_ds, "test")

        print(f"\nMisclassified files copied into: {self.post_analysis_dir}/")
        print("*" * 50)

# ----------------------------------------------------------------------------
# Example usage:
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Below are two sample trainers: one for "resemblyzer-embeddings.jsonl"
    with dimension 256, and another for "ecapa-embeddings.jsonl" with dimension 192.
    You can comment/uncomment whichever you need.
    """

    # Example 1: Train on Resemblyzer embeddings
    resemblyzer_trainer = FNNTrainer(
        jsonl_path="models-embeddings/resemblyzer-embeddings.jsonl",
        model_name="resemblyzer",
        embedding_dim=256,   # resemblyzer is typically 256-d
        hidden_dim=256,
        batch_size=2048,
        lr=1e-3,
        dropout=0.5,
        patience=10,
        max_epochs=1000,
        seed=42
    )
    resemblyzer_trainer.train()
    resemblyzer_trainer.post_analysis()

    # Example 2: Train on ECAPA embeddings
    ecapa_trainer = FNNTrainer(
        jsonl_path="models-embeddings/ecapa-embeddings.jsonl",
        model_name="ecapa",
        embedding_dim=192,  # ecapa is typically 192-d
        hidden_dim=192,
        batch_size=2048,
        lr=1e-3,
        dropout=0.5,
        patience=10,
        max_epochs=1000,
        seed=42
    )
    ecapa_trainer.train()
    ecapa_trainer.post_analysis()
