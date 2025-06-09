import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import os
import random
from tqdm import tqdm

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# Paths & constants
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures_VGG')
models_path = os.path.join(home_path, 'reports', 'models')
os.makedirs(figures_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Data
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)

# Utils
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            pred_labels = preds.argmax(dim=1)
            correct += (pred_labels == y).sum().item()
            total += y.size(0)
    return correct / total

def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=20):
    model.to(device)
    batches_n = len(train_loader)
    losses_list = []
    grads = []
    val_acc_list = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()
        loss_list = []
        grad_list = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            # 记录最后一层的梯度范数
            grad_norm = None
            for name, param in model.named_parameters():
                if 'weight' in name and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    break
            grad_list.append(grad_norm)
            optimizer.step()
            loss_list.append(loss.item())
        losses_list.append(loss_list)
        grads.append(grad_list)
        val_acc = get_accuracy(model, val_loader)
        val_acc_list.append(val_acc)
        print(f"Epoch {epoch}: val acc={val_acc:.4f}, loss mean={np.mean(loss_list):.4f}")
    return losses_list, grads, val_acc_list

# Loss Landscape 可视化
def plot_loss_landscape(result_dict, figures_path):
    plt.figure(figsize=(14, 8))
    for (model_name, lr), result in result_dict.items():
        epochs = np.arange(len(result['min_curve']))
        label = f"{model_name} lr={lr}"
        plt.fill_between(epochs, result['min_curve'], result['max_curve'], alpha=0.18)
        plt.plot(epochs, result['min_curve'], '--', label=f"Min {label}", linewidth=1)
        plt.plot(epochs, result['max_curve'], '-', label=f"Max {label}", linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Landscape Bounds Across Epochs (BN vs Non-BN, Multiple LRs)')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'loss_landscape_compare.png'))
    plt.close()

# （可选）梯度变化可视化
def plot_gradient_norms(result_dict, figures_path):
    plt.figure(figsize=(14, 8))
    for (model_name, lr), result in result_dict.items():
        epochs = np.arange(len(result['grads']))
        label = f"{model_name} lr={lr}"
        plt.plot(epochs, result['grads'], label=f"Grad Norm {label}", linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norms Across Epochs (BN vs Non-BN)')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'gradient_norms_compare.png'))
    plt.close()

def main():
    # 多学习率与BN/非BN对比
    learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]
    model_types = {'No_BN': VGG_A, 'With_BN': VGG_A_BatchNorm}
    epochs = 20

    result_dict = {}

    for model_name, model_cls in model_types.items():
        for lr in learning_rates:
            print(f"Training [{model_name}] with learning rate={lr}")
            set_random_seeds(2020, device=device)
            model = model_cls()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            losses, grads, val_acc_list = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epochs)
            losses_np = np.array(losses)
            min_curve = np.min(losses_np, axis=1)
            max_curve = np.max(losses_np, axis=1)
            # 保存单组数据
            np.savetxt(os.path.join(figures_path, f'loss_{model_name}_lr{lr}.txt'), np.mean(losses_np, axis=1), fmt='%0.4f')
            np.savetxt(os.path.join(figures_path, f'min_curve_{model_name}_lr{lr}.txt'), min_curve, fmt='%0.4f')
            np.savetxt(os.path.join(figures_path, f'max_curve_{model_name}_lr{lr}.txt'), max_curve, fmt='%0.4f')
            np.savetxt(os.path.join(figures_path, f'grad_{model_name}_lr{lr}.txt'), [np.mean(g) for g in grads], fmt='%0.4f')
            result_dict[(model_name, lr)] = {
                'min_curve': min_curve,
                'max_curve': max_curve,
                'mean_curve': np.mean(losses_np, axis=1),
                'val_acc': val_acc_list,
                'grads': [np.mean(g) for g in grads]
            }
            plot_loss_landscape(result_dict, figures_path)
            plot_gradient_norms(result_dict, figures_path)

if __name__ == "__main__":
    main()
    