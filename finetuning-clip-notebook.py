# CLIP Fine-tuning for Image Classification

# Import required libraries
import clip
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
import wandb

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", jit=False)
model.to(device)

# Load and preprocess dataset
dataset = load_dataset("rootsautomation/RICO-SCA")
dataset = dataset['train'].select_columns(['image', 'category'])
dataset = dataset.select(range(50000))  # Limit to 50,000 samples for faster processing

# Get unique categories
categories = list(set(example['category'] for example in dataset))
num_classes = len(categories)

# Split dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Custom dataset class
class ScreenshotsDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image']
        category = item['category']
        label = categories.index(category)
        return self.transform(image), label

# Create DataLoaders
train_loader = DataLoader(ScreenshotsDataset(train_dataset), batch_size=32, shuffle=True)
val_loader = DataLoader(ScreenshotsDataset(val_dataset), batch_size=32, shuffle=False)

# Fine-tuning model class
class CLIPFineTuner(nn.Module):
    def __init__(self, model, num_classes):
        super(CLIPFineTuner, self).__init__()
        self.model = model
        self.classifier = nn.Linear(model.visual.output_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float()
        return self.classifier(features)

# Initialize fine-tuning model
model_ft = CLIPFineTuner(model, num_classes).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ft.classifier.parameters(), lr=1e-4)

# Compute metrics function
def compute_metrics(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = 100 * correct / total
    f1 = f1_score(labels.cpu(), predicted.cpu(), average='weighted')
    return accuracy, f1, predicted.cpu().numpy(), labels.cpu().numpy()

# Initialize wandb
wandb.init(project="finetuning_for_classification", name="main_run")
wandb.config.update({
    "learning_rate": optimizer.param_groups[0]['lr'],
    "batch_size": train_loader.batch_size,
    "num_epochs": 10,
})

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model_ft.train()
    running_loss = 0.0
    running_corrects = 0
    all_labels = []
    all_predictions = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Loss: 0.0000")

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_ft(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(preds.cpu().numpy())

        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # Compute and log training metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    epoch_f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, F1: {epoch_f1:.4f}')

    wandb.log({
        "epoch": epoch,
        "train_loss": epoch_loss,
        "train_accuracy": epoch_acc,
        "train_f1": epoch_f1,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

    # Validation
    model_ft.eval()
    val_loss = 0.0
    val_corrects = 0
    all_val_labels = []
    all_val_predictions = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_ft(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            accuracy, f1, predictions, true_labels = compute_metrics(outputs, labels)
            val_corrects += accuracy * labels.size(0) / 100
            all_val_labels.extend(true_labels)
            all_val_predictions.extend(predictions)

    # Compute and log validation metrics
    val_loss /= len(val_loader)
    val_accuracy = val_corrects / len(val_loader.dataset)
    val_f1 = f1_score(all_val_labels, all_val_predictions, average='weighted')
    
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}')

    cm = confusion_matrix(all_val_labels, all_val_predictions)
    
    wandb.log({
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_f1": val_f1,
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_val_labels,
            preds=all_val_predictions,
            class_names=categories
        )
    })

# Save the fine-tuned model
torch.save(model_ft.state_dict(), 'clip_finetuned.pth')

# Finish the wandb run
wandb.finish()
