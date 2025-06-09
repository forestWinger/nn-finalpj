from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from PIL import Image
import yaml

def pil_to_rgb(img):
    return img.convert('RGB')

def round_list(tensor, digits=3):
    return [round(x, digits) for x in tensor.tolist()]

def compute_mean_std(dataset):
    """
    Compute the mean and standard deviation of a dataset.
    Args:
        dataset: A PyTorch dataset.
    Returns:
        mean: A tensor containing the mean of each channel.
        std: A tensor containing the standard deviation of each channel.
    """
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    n_images = 0
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)

    for images, _ in tqdm(loader):
        n_images += images.size(0)
        channel_sum += images.sum(dim=[0, 2, 3])  # sum over batch, height, width
        channel_squared_sum += (images ** 2).sum(dim=[0, 2, 3])

    # Calculate mean and std
    mean = channel_sum / (n_images * 32 * 32)
    std = (channel_squared_sum / (n_images * 32 * 32) - mean ** 2).sqrt()

    return mean, std

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Lambda(pil_to_rgb),
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)
    train_mean, train_std = compute_mean_std(trainset)
    test_mean, test_std = compute_mean_std(testset)
    print(f"CIFAR-10 's trainset mean: {train_mean}")
    print(f"CIFAR-10 's trainset std:  {train_std}")

    print(f"CIFAR-10 's testset mean: {test_mean}")
    print(f"CIFAR-10 's testset std:  {test_std}")

    # save the mean and std to a YAML file
    with open('./configs/cifar10_mean_std.yaml', 'w') as f:
        yaml.dump({'train': {'mean': round_list(train_mean), 'std': round_list(train_std)},
                    'test': {'mean': round_list(test_mean), 'std': round_list(test_std)}}, f)

