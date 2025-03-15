import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import timm

class TrainableMemory(nn.Module):
    """
    A neural memory module with differentiable operations, learnable parameters,
    and a similarity threshold for filtering predictions.
    """
    def __init__(self, feature_dim, num_classes, memory_size=1000, temperature=0.1, similarity_threshold=0.8):
        super(TrainableMemory, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.memory_size = memory_size
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold  
        
        # Learnable parameters (initialized with zeros)
        self.memory_keys = nn.Parameter(torch.zeros(memory_size, feature_dim)) # Features
        self.memory_values = nn.Parameter(torch.zeros(memory_size, num_classes)) # Classes
        self.is_full = False
        
        # Initialize with small noise
        nn.init.normal_(self.memory_keys, mean=0.0, std=0.01)
        nn.init.normal_(self.memory_values, mean=0.0, std=0.01)

            
    def retrieve(self, query_features):
        """
        Retrieve nearest neighbors using cosine similarity.
        Returns predicted classes and confidence scores.
        """
        batch_size = query_features.size(0)
        device = query_features.device
        
        # If memory is empty, return zeros with no confidence
        if len(self.memory_keys) == 0:
            return (
                torch.zeros(batch_size, dtype=torch.long, device=device),
                torch.zeros(batch_size, device=device)
            )
        
        memory_keys = self.memory_keys.to(device)
        memory_values = self.memory_values.to(device)
        
        # Normalize vectors for cosine similarity
        query_normalized = F.normalize(query_features, p=2, dim=1)
        memory_normalized = F.normalize(memory_keys, p=2, dim=1)
        
        # Compute cosine similarity
        similarities = torch.matmul(query_normalized, memory_normalized.t())
        
        # Get nearest neighbor and its similarity score
        confidence_scores, indices = similarities.max(dim=1)

        # Get the index of the max value in the one-hot encoded memory_values
        retrieved_classes = torch.argmax(memory_values[indices], dim=1)
        
        return retrieved_classes, confidence_scores


    
    def memory_contrastive_loss(self, query_features, labels):
        """
        Differentiable contrastive loss that directly optimizes memory contents.
        """
        # Normalize vectors
        query_norm = F.normalize(query_features, p=2, dim=1)
        key_norm = F.normalize(self.memory_keys, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(query_norm, key_norm.t())  # [B, M]
        
        # Create positive mask
        pos_mask = torch.eq(labels.unsqueeze(1), 
                            torch.argmax(self.memory_values, dim=1).unsqueeze(0))
        
        # Calculate contrastive loss
        logits = sim_matrix / self.temperature
        loss = F.cross_entropy(logits, pos_mask.float().argmax(dim=1))
        
        return loss

    # TrainableMemory update method modification
    def update(self, features, labels, update_rate=0.25):
        """
        Updates the trainable memory with new examples.
        If memory is full, replaces low confidence items with new examples.

        Args:
            features: Input feature vectors [batch_size, feature_dim]
            labels: Target class labels [batch_size]
            update_rate: Controls how quickly memory adapts to new examples
        """
        batch_size = features.size(0)
        device = features.device

        # If no examples to update with, return early
        if batch_size == 0:
            return

        # Move memory to same device as input if necessary
        if self.memory_keys.device != device:
            self.memory_keys.data = self.memory_keys.data.to(device)
            self.memory_values.data = self.memory_values.data.to(device)

        # Convert labels to one-hot encoding
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float()

        if not self.is_full:
            # If memory not full, find unused slots (initialized slots have low norm)
            key_norms = torch.norm(self.memory_keys, p=2, dim=1)
            # Find indices of slots with smallest norm (likely unused)
            _, unused_indices = torch.topk(key_norms, k=min(batch_size, self.memory_size),
                                        largest=False)

            # Update these memory slots with new examples
            for i, idx in enumerate(unused_indices[:batch_size]):
                # Gradually update the memory keys and values
                self.memory_keys.data[idx] = (1 - update_rate) * self.memory_keys.data[idx] + update_rate * features[i]
                self.memory_values.data[idx] = (1 - update_rate) * self.memory_values.data[idx] + update_rate * one_hot_labels[i]
        else:
            # Memory is full - identify low confidence items to replace
            
            # Normalize memory keys for similarity calculation
            memory_normalized = F.normalize(self.memory_keys, p=2, dim=1)
            
            # For each new example, find its similarity with all memory items
            for i in range(batch_size):
                query_normalized = F.normalize(features[i].unsqueeze(0), p=2, dim=1)
                
                # Compute similarities with all memory items
                similarities = torch.matmul(query_normalized, memory_normalized.t())
                
                # Get maximum similarity
                max_similarity, _ = similarities.max(dim=1)
                
                # If the maximum similarity is below threshold, replace the least confident memory item
                if max_similarity < self.similarity_threshold:
                    # Find the memory item with lowest confidence
                    # Confidence here is measured by the max value in memory_values (certainty of class prediction)
                    confidence_scores, _ = torch.max(self.memory_values, dim=1)
                    
                    # Get index of lowest confidence item
                    min_confidence_idx = torch.argmin(confidence_scores)
                    
                    # Replace with new item
                    self.memory_keys.data[min_confidence_idx] = features[i]
                    self.memory_values.data[min_confidence_idx] = one_hot_labels[i]

        
        if len(self.memory_keys) >= self.memory_size:
            self.is_full = True

 
class CachedMemory(nn.Module):
    """
    A simplified trainable memory module that stores feature vectors and their labels.
    Uses direct similarity comparison
    """
    def __init__(self, feature_dim, num_classes, memory_size=1000, similarity_threshold=0.8):
        super(CachedMemory, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.memory_size = memory_size
        self.similarity_threshold = similarity_threshold
        
        # Initialize memory structures
        self.memory_keys = torch.zeros(0, feature_dim)  # Feature vectors
        self.memory_values = torch.zeros(0, dtype=torch.long)  # Class labels
        self.is_full = False
        
    def retrieve(self, query_features):
        """
        Retrieve nearest neighbors using cosine similarity.
        Returns predicted classes and confidence scores.
        """
        batch_size = query_features.size(0)
        device = query_features.device
        
        # If memory is empty, return zeros with no confidence
        if len(self.memory_keys) == 0:
            return (
                torch.zeros(batch_size, dtype=torch.long, device=device),
                torch.zeros(batch_size, device=device)
            )
        
        memory_keys = self.memory_keys.to(device)
        memory_values = self.memory_values.to(device)
        
        # Normalize vectors for cosine similarity
        query_normalized = F.normalize(query_features, p=2, dim=1)
        memory_normalized = F.normalize(memory_keys, p=2, dim=1)
        
        # Compute cosine similarity
        similarities = torch.matmul(query_normalized, memory_normalized.t())
        
        # Get nearest neighbor and its similarity score
        confidence_scores, indices = similarities.max(dim=1)
        retrieved_classes = memory_values[indices]
        
        return retrieved_classes, confidence_scores
        
    def update(self, features, labels):
        """
        Update memory with new examples.
        If memory is full, replace low confidence items based on similarity threshold.
        """
        batch_size = features.size(0)
        device = features.device
        
        # Handle empty batch
        if batch_size == 0:
            return
            
        # Process items to add to memory
        features = features.detach().cpu()
        labels = labels.detach().cpu()
        
        if not self.is_full:
            # If memory not full, add new items until full
            available_space = self.memory_size - len(self.memory_keys)
            
            if available_space == 0:
                self.is_full = True
            else:
                # Add as many examples as we have space for
                num_to_add = min(batch_size, available_space)
                
                # Add new items to memory
                self.memory_keys = torch.cat([self.memory_keys, features[:num_to_add]], dim=0)
                self.memory_values = torch.cat([self.memory_values, labels[:num_to_add]], dim=0)
                
                if len(self.memory_keys) >= self.memory_size:
                    self.is_full = True
                    
                # Process remaining items if we've filled memory
                features = features[num_to_add:]
                labels = labels[num_to_add:]
                batch_size = len(features)
        
        # If memory is full and we still have items to process
        if self.is_full and batch_size > 0:
            # For memory items on CPU, need to process them on the same device
            memory_keys_device = self.memory_keys.to(device)
            memory_values_device = self.memory_values.to(device)
            features_device = features.to(device)
            
            # Normalize vectors for cosine similarity
            memory_normalized = F.normalize(memory_keys_device, p=2, dim=1)
            
            # For each remaining feature, check if it should replace an existing memory item
            for i in range(batch_size):
                feature = features_device[i].unsqueeze(0)  # Add batch dimension
                label = labels[i].item()
                
                # Normalize feature
                feature_normalized = F.normalize(feature, p=2, dim=1)
                
                # Compute similarity with all memory items
                similarities = torch.matmul(feature_normalized, memory_normalized.t()).squeeze()
                
                # Get maximum similarity
                max_similarity, max_idx = similarities.max(dim=0)
                
                # If max similarity is below threshold, replace that memory item
                if max_similarity < self.similarity_threshold:
                    # Find memory items with the same class label (if any)
                    same_class_indices = (memory_values_device == label).nonzero(as_tuple=True)[0]
                    
                    if len(same_class_indices) > 0:
                        # If there are memory items of the same class, replace the one with lowest similarity
                        same_class_similarities = similarities[same_class_indices]
                        lowest_similarity_idx = same_class_indices[torch.argmin(same_class_similarities)]
                        
                        # Replace on CPU
                        self.memory_keys[lowest_similarity_idx] = features[i]
                        self.memory_values[lowest_similarity_idx] = labels[i]
                    else:
                        # Otherwise replace the memory item with overall lowest similarity
                        lowest_similarity_idx = torch.argmin(similarities)
                        
                        # Replace on CPU
                        self.memory_keys[lowest_similarity_idx] = features[i]
                        self.memory_values[lowest_similarity_idx] = labels[i]
    
    def memory_contrastive_loss(self, features, labels, temperature=0.1):
        """
        Compute contrastive loss between current batch features and memory items.
        More robust to batch size mismatches and edge cases.
        """
        if len(self.memory_keys) == 0:
            return torch.tensor(0.0, device=features.device)
        
        batch_size = features.size(0)  # Get actual batch size
        device = features.device
        
        
        memory_keys = self.memory_keys.to(device)
        memory_values = self.memory_values.to(device)
        
        # Ensure batch dimensions match for all tensors
        if batch_size == 0:
            return torch.tensor(0.0, device=device)
        
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=1)
        memory_keys_norm = F.normalize(memory_keys, p=2, dim=1)
        
        # Compute similarities [batch_size, memory_size]
        similarities = torch.matmul(features_norm, memory_keys_norm.t()) / temperature
        
        # Create positive mask with proper broadcasting
        # [batch_size, 1] == [1, memory_size] -> [batch_size, memory_size]
        positive_mask = (labels.unsqueeze(1) == memory_values.unsqueeze(0))
        positive_mask = positive_mask.float()
        
        # Compute exp(similarities) with same shape as positive_mask
        exp_similarities = torch.exp(similarities)  # [batch_size, memory_size]
        
        # Element-wise multiplication
        pos_similarities = exp_similarities * positive_mask  # [batch_size, memory_size]
        
        # Sum along memory dimension
        numerator = pos_similarities.sum(dim=1)  # [batch_size]
        denominator = exp_similarities.sum(dim=1)  # [batch_size]
        
        # Handle valid samples
        valid_samples = (positive_mask.sum(dim=1) > 0)
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        # Compute loss only for valid samples
        losses = -torch.log(numerator / denominator)
        loss = losses[valid_samples].mean()
        
        return loss

class EnhancedClassifier(nn.Module):
    """
    Enhanced classifier with configurable memory module and dropout regularization.
    """
    def __init__(self, backbone, feature_dim, num_classes, memory_module_class,
                 memory_size=1000, confidence_threshold=0.8, dropout_rate=0.2):
        super(EnhancedClassifier, self).__init__()
        self.backbone = backbone
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate), 
            nn.Linear(feature_dim, num_classes)
        )
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        
        self.memory = memory_module_class(
            feature_dim=feature_dim,
            num_classes=num_classes,
            memory_size=memory_size,
            similarity_threshold=confidence_threshold
        )
        
        # Track usage statistics
        self.memory_hits = 0
        self.nn_usage = 0
        
        # Store memory type for plotting
        self.memory_type = memory_module_class.__name__

    
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        batch_size = features.size(0)
        device = x.device
        
        memory_classes, confidence_scores = self.memory.retrieve(features)
        
        memory_logits = torch.zeros(batch_size, self.num_classes, device=device)
        for i in range(batch_size):
            memory_logits[i, memory_classes[i]] = 1.0
        
        nn_logits = self.classifier(features)
        
        final_logits = torch.zeros_like(nn_logits)
        sources = []
        
        for i in range(batch_size):
            if confidence_scores[i] >= self.confidence_threshold:
                final_logits[i] = memory_logits[i]
                sources.append("memory")
                self.memory_hits += 1
            else:
                final_logits[i] = nn_logits[i]
                sources.append("network")
                self.nn_usage += 1
        
        if return_features:
            return final_logits, features, sources
        return final_logits, sources



def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    correct = 0
    total = 0
    memory_hits = 0
    nn_usage = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits, _, sources = model(inputs, return_features=True)
            
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            memory_hits += sources.count("memory")
            nn_usage += sources.count("network")
    
    accuracy = 100. * correct / total
    memory_usage = 100. * memory_hits / total
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Memory usage: {memory_usage:.2f}%")
    print(f"Memory hits: {memory_hits}, NN usage: {nn_usage}")

def get_cifar10_loaders(batch_size=64, num_workers=2):
    """Load and prepare CIFAR-10 dataset with more augmentations"""
    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Add color jitter
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    # CIFAR-10 classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes



def run_cifar10_experiment(memory_module_class, memory_size=5000, confidence_threshold=0.85, 
                          num_epochs=15, batch_size=128, memory_loss_weight=0.2,model_name=None):
    """
    Run a complete experiment with the enhanced memory-based classifier on CIFAR-10.
    Args:
        memory_module_class: Class to use for memory module (TrainableMemory or CachedMemory)
    """
    memory_type = memory_module_class.__name__
    print(f"Running Enhanced CIFAR-10 experiment with {memory_type}:")
    print(f"- Memory size: {memory_size}")
    print(f"- Confidence threshold: {confidence_threshold}")
    print(f"- Number of epochs: {num_epochs}")
    print(f"- Batch size: {batch_size}")
    print(f"- Memory loss weight: {memory_loss_weight}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader, classes = get_cifar10_loaders(batch_size=batch_size)
    
    # Create model with TIMM backbone
    print(f"Creating enhanced classifier with {memory_type}...")
    backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
    feature_dim = backbone.num_features
    
    model = EnhancedClassifier(
        backbone=backbone,
        feature_dim=feature_dim,
        num_classes=10,
        memory_module_class=memory_module_class,
        memory_size=memory_size,
        confidence_threshold=confidence_threshold
    )
    model = model.to(device)
    
    # Training metrics
    train_losses = []
    validation_losses = []
    memory_losses = []
    train_accuracies = []
    test_accuracies = []
    memory_usage_percents = []
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.001,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.1
    )
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_memory_loss = 0.0
        correct = 0
        total = 0
        

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logits, features, _ = model(inputs, return_features=True)
            
            # Classification loss
            cls_loss = criterion(logits, labels)
            
            # Memory contrastive loss
            memory_loss = model.memory.memory_contrastive_loss(features, labels)
            
            # Total loss
            total_loss = (1.0-memory_loss_weight) * cls_loss + memory_loss_weight * memory_loss
            
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()
            
            model.memory.update(features.detach(), labels)
            
            # Statistics
            running_loss += total_loss.item()
            running_memory_loss += memory_loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': running_loss / (train_pbar.n + 1),
                'acc': 100. * correct / total,
                'mem_loss': running_memory_loss / (train_pbar.n + 1)
            })
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        memory_loss = running_memory_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        
        train_losses.append(train_loss)
        memory_losses.append(memory_loss)
        train_accuracies.append(train_accuracy)
        
        # Evaluation phase
        model.eval()
        test_correct = 0
        test_total = 0
        memory_hits = 0
        nn_usage = 0
        
        test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]")
        with torch.no_grad():
            val_total_loss = 0.0
            for inputs, labels in test_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                
                logits, features, sources = model(inputs, return_features=True)

                # Calculate validation loss
                val_cls_loss = criterion(logits, labels)
                
                val_memory_loss = model.memory.memory_contrastive_loss(features, labels)
                val_total_loss += ((1.0-memory_loss_weight) * val_cls_loss + memory_loss_weight * val_memory_loss).item()
                
                # Calculate accuracy
                _, predicted = logits.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                
                # Track memory usage
                memory_hits += sources.count("memory")
                nn_usage += sources.count("network")
                
                # Update progress bar
                test_pbar.set_postfix({
                    'acc': 100. * test_correct / test_total,
                    'mem%': 100. * memory_hits / test_total
                })
            validation_losses.append(val_total_loss / len(test_loader))    
        
        # Calculate test metrics
        test_accuracy = 100. * test_correct / test_total
        memory_usage = 100. * memory_hits / test_total
        
        test_accuracies.append(test_accuracy)
        memory_usage_percents.append(memory_usage)
        
        
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Memory Loss: {memory_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.2f}%")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print(f"Memory Usage: {memory_usage:.2f}%")
        print(f"Memory Size: {len(model.memory.memory_keys)}/{memory_size}")
        if model.memory.is_full:
            print("Memory is full")
        print("-" * 80)
    
    plt.figure(figsize=(20, 5))
    
    plt.subplot(141)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.title(f'{memory_type} - Total Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(142)
    plt.plot(memory_losses, label='Memory Loss')
    plt.title(f'{memory_type} - Memory Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(143)
    plt.plot(train_accuracies, label='Train')
    plt.plot(test_accuracies, label='Test')
    plt.title(f'{memory_type} - Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.subplot(144)
    plt.plot(memory_usage_percents, label='Memory Usage')
    plt.title(f'{memory_type} - Memory Usage vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Memory Usage (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'memory_classifier_metrics_{memory_type.lower()}.png')
    plt.close()
    
    torch.save(model.state_dict(), f'memory_classifier_{memory_type.lower()}.pth')
    print(f"Model saved as 'memory_classifier_{memory_type.lower()}.pth'")
    
    return {
        'train_losses': train_losses,
        'validation_losses': validation_losses,
        'memory_losses': memory_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'memory_usage_percents': memory_usage_percents,
        'final_test_accuracy': test_accuracies[-1],
        'final_memory_usage': memory_usage_percents[-1]
    }


def evaluate_inference_speed(model, test_loader, device, num_runs=3):
    """
    Evaluate inference speed comparing memory vs. neural network paths.
    """
    print("\nEvaluating inference speed...")
    

    model = model.to(device)
    
    model.eval()
    memory_times = []
    nn_times = []
    
    # Warmup run
    print("Performing warmup runs...")
    with torch.no_grad():
        for inputs, _ in [next(iter(test_loader))]:
            inputs = inputs.to(device)
            features = model.backbone(inputs)
            _ = model.memory.retrieve(features)
            _ = model.classifier(features)
    
    # Timing runs
    print(f"Running {num_runs} timing passes...")
    with torch.no_grad():
        for run in range(num_runs):
            print(f"Timing run {run+1}/{num_runs}...")
            
            for inputs, _ in tqdm(test_loader, desc="Measuring inference time"):
                inputs = inputs.to(device)
                batch_size = inputs.size(0)
                
                # Time feature extraction (common to both paths)
                torch.cuda.synchronize()
                start_features = time.time()
                features = model.backbone(inputs)
                torch.cuda.synchronize()
                feature_time = time.time() - start_features
                
                # Time memory retrieval
                torch.cuda.synchronize()
                start_memory = time.time()
                _ = model.memory.retrieve(features)
                torch.cuda.synchronize()
                memory_time = time.time() - start_memory
                
                # Time neural network
                torch.cuda.synchronize()
                start_nn = time.time()
                _ = model.classifier(features)
                torch.cuda.synchronize()
                nn_time = time.time() - start_nn
                
                # Record times (per sample)
                memory_times.append((memory_time + feature_time) / batch_size)
                nn_times.append((nn_time + feature_time) / batch_size)
    
    # Calculate average times
    avg_memory_time = sum(memory_times) / len(memory_times)
    avg_nn_time = sum(nn_times) / len(nn_times)
    
    print(f"\nAverage memory path time: {avg_memory_time*1000:.2f} ms per sample")
    print(f"Average neural network path time: {avg_nn_time*1000:.2f} ms per sample")
    
    if avg_memory_time < avg_nn_time:
        print(f"Memory path is {avg_nn_time/avg_memory_time:.2f}x faster than neural network")
    else:
        print(f"Memory path is {avg_memory_time/avg_nn_time:.2f}x slower than neural network")
    
    return {
        'avg_memory_time': avg_memory_time,
        'avg_nn_time': avg_nn_time,
        'speedup': avg_nn_time/avg_memory_time if avg_memory_time < avg_nn_time else -avg_memory_time/avg_nn_time
    }


def run_trainable_memory_train_and_test(
        memory_size = 10000,
        batch_size = 128,
        confidence_threshold = 0.85,
        num_epochs = 5,
        memory_loss_weight = 0.4,
        model_name = 'resnet34'
    ):

    metrics_trainable = run_cifar10_experiment(
        memory_module_class=TrainableMemory,
        memory_size=memory_size,
        confidence_threshold=confidence_threshold,
        num_epochs=num_epochs,
        batch_size=batch_size,
        memory_loss_weight=memory_loss_weight,
        model_name = model_name
    )
    
    print("\nTrainable Memory Results:")
    print(f"Test Accuracy: {metrics_trainable['final_test_accuracy']:.2f}%")
    print(f"Memory Usage: {metrics_trainable['final_memory_usage']:.2f}%")
    
    backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
    model = EnhancedClassifier(
        memory_module_class=TrainableMemory,        
        backbone=backbone,
        feature_dim=backbone.num_features,
        num_classes=10,
        memory_size=memory_size,
        confidence_threshold=confidence_threshold
    )
    
    model.load_state_dict(torch.load('memory_classifier_trainablememory.pth'))
        
    _, test_loader, _ = get_cifar10_loaders(batch_size=128)
    
    speed_metrics = evaluate_inference_speed(model, test_loader,                                                                                        
                                          torch.device('cuda' if torch.cuda.is_available() else 'cpu'))    
    

def run_cached_memory_train_and_test(
        memory_size = 10000,
        batch_size = 128,
        confidence_threshold = 0.85,
        num_epochs = 5,
        memory_loss_weight = 0.4,
        model_name = 'resnet34'
    ):

    metrics_cached = run_cifar10_experiment(
        memory_module_class=CachedMemory,
        memory_size=memory_size,
        confidence_threshold=confidence_threshold,
        num_epochs=num_epochs,
        batch_size=batch_size,
        memory_loss_weight=memory_loss_weight,
        model_name = model_name
    )
    
    print("\nCached Memory Results:")
    print(f"Test Accuracy: {metrics_cached['final_test_accuracy']:.2f}%")
    print(f"Memory Usage: {metrics_cached['final_memory_usage']:.2f}%")

    backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
    model = EnhancedClassifier(
        memory_module_class=CachedMemory,        
        backbone=backbone,
        feature_dim=backbone.num_features,
        num_classes=10,
        memory_size=memory_size,
        confidence_threshold=confidence_threshold
    )
    
    model.load_state_dict(torch.load('memory_classifier_cachedmemory.pth'))
        
    _, test_loader, _ = get_cifar10_loaders(batch_size=128)
    
    speed_metrics = evaluate_inference_speed(model, test_loader,                                                                                        
                                          torch.device('cuda' if torch.cuda.is_available() else 'cpu'))    
    

if __name__ == "__main__":
    memory_size = 10000
    batch_size = 128
    confidence_threshold = 0.85
    num_epochs = 5
    memory_loss_weight = 0.4
    model_name = 'resnet50'
    run_cached_memory_train_and_test(
        memory_size,
        batch_size,
        confidence_threshold,
        num_epochs,
        memory_loss_weight,
        model_name
        )
    run_trainable_memory_train_and_test(
        memory_size,
        batch_size,
        confidence_threshold,
        num_epochs,
        memory_loss_weight,
        model_name        
    )
