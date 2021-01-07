import torch
import torchvision


def KMNIST(batch_size):
    """
    Function returning a train, validation, and test loader of the KMNIST
    dataset.
    The validation set is based on 10000 random instances of the train set.

    Args:
        batch_size (int): Amount of instances a batch returned by each loader
            should have.

    Returns:
        {
            "train": Batch loader of the train set,
            "val": Batch loader of the val set,
            "test": Batch loader of the test set
        }
    """

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )

    # get original train set
    set = torchvision.datasets.KMNIST(
        root='./data', train=True, download=True, transform=transform
    )

    train_set, val_set = torch.utils.data.random_split(
        set, [50000, 10000]
    )
    test_set = torchvision.datasets.KMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
