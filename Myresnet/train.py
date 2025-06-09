import torch as th
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import model
from PIL import Image
import argparse
import yaml
import os
import time

def pil_to_rgb(img):
    return img.convert('RGB')

def train():
    """
    Train a ResNet-18 model on the CiFAR-10 dataset.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/config_1.yaml')
    args = parser.parse_args()
    config_basename = os.path.splitext(os.path.basename(args.config))[0]
    print(f"Using config: {config_basename}")

    writer = SummaryWriter(log_dir=f"runs/{config_basename}/" + time.strftime("%Y%m%d-%H%M%S"))
    print(f"TensorBoard logs will be saved to: {writer.log_dir}")

    with open(args.config, 'r') as f:
        params = yaml.safe_load(f)
        print(f"Parameters: {params}")

    myresnet = model.resnet18()

    fc_params = list(myresnet.fc.parameters())
    backbone_params = [p for name, p in myresnet.named_parameters() if "fc" not in name]

    optimizer = th.optim.SGD([
        {"params": backbone_params, "lr": params["learning_rate"]},     
        {"params": fc_params, "lr": params["learning_rate"]}                
    ], momentum=params["momentum"], weight_decay=params["weight_decay"])

    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Training in GPU
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    myresnet.to(device)

    dummy_input = th.randn(1, 3, 32, 32).to(device)
    writer.add_graph(myresnet, dummy_input)

    # load data
    data_root = './data'

    transform_train = transforms.Compose([
        transforms.Lambda(pil_to_rgb),
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),  # CIFAR-10 images are 32x32
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.447],  # CIFAR-10 dataset mean
                            std=[0.247, 0.243, 0.262])    # CIFAR-10 dataset std
    ])

    transform_test = transforms.Compose([
        transforms.Lambda(pil_to_rgb),
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616])
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = th.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    val_loader = th.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    os.makedirs('./saved_models', exist_ok=True)
    # Training loop
    best_val_loss = float('inf')
    patience = 10  # Early stopping patience
    for epoch in range(params["num_epochs"]):
        myresnet.train()
        train_loss_sum = 0.0
        train_samples = 0
        train_correct = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = myresnet(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item() * inputs.size(0)
            train_samples += inputs.size(0)
            _, preds = outputs.max(1)
            train_correct += (preds == labels).sum().item()

        avg_train_loss = train_loss_sum / train_samples
        train_accuracy = train_correct / train_samples

        # Validation
        myresnet.eval()
        val_loss_sum = 0.0
        val_samples = 0
        val_correct = 0
        with th.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = myresnet(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                val_loss_sum += loss.item() * inputs.size(0)
                val_samples += inputs.size(0)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
        avg_val_loss = val_loss_sum / val_samples
        val_accuracy = val_correct / val_samples

        scheduler.step(val_accuracy)

        print(f"Epoch [{epoch+1}/{params['num_epochs']}], "
            f"TrainLoss: {avg_train_loss:.4f}, TrainAcc: {train_accuracy:.4f}, "
            f"ValLoss: {avg_val_loss:.4f}, ValAcc: {val_accuracy:.4f}")
        
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            th.save(myresnet.state_dict(), f'./saved_models/best_model_{config_basename}.pth')
        else:
            trigger_times += 1
            print(f'No improvement. Early stopping trigger: {trigger_times}/{patience}')
        if trigger_times >= patience:
            print('Early stopping!')
            break
    
    writer.close()
        
if __name__ == "__main__":
    train()