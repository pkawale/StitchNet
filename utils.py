import os
import certifi
import logging
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def setup_logging(model1, model2):
    logging.basicConfig(level=logging.INFO)
    training_logger = logging.getLogger("training")
    testing_logger = logging.getLogger("testing")
    comparison_logger = logging.getLogger("comparison")

    training_handler = logging.FileHandler(f"training_{model1}_{model2}.log")
    testing_handler = logging.FileHandler(f"testing_{model1}_{model2}.log")
    comparison_handler = logging.FileHandler(f"comparison_{model1}_{model2}.log")

    training_handler.setLevel(logging.INFO)
    testing_handler.setLevel(logging.INFO)
    comparison_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    training_handler.setFormatter(formatter)
    testing_handler.setFormatter(formatter)
    comparison_handler.setFormatter(formatter)

    training_logger.addHandler(training_handler)
    testing_logger.addHandler(testing_handler)
    comparison_logger.addHandler(comparison_handler)

    return training_logger, testing_logger, comparison_logger


def load_dataset(batch_size=64, num_workers=4, pin_memory=True):
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),  # CIFAR-10 image size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    cifar10_train = datasets.CIFAR10(
        root=os.path.join(os.getenv("DATA_DIR", "data"), "cifar10"),
        train=True,
        download=True,
        transform=transform,
    )
    cifar10_test = datasets.CIFAR10(
        root=os.path.join(os.getenv("DATA_DIR", "data"), "cifar10"),
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        cifar10_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        cifar10_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


