from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, STL10


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    num_classes: int
    dataset_cls: type
    train_kwargs: dict[str, object]
    eval_train_kwargs: dict[str, object]
    eval_test_kwargs: dict[str, object]
    target_attr: str


DATASET_REGISTRY = {
    'cifar10': DatasetMetadata(
        name='cifar10',
        image_size=32,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
        num_classes=10,
        dataset_cls=CIFAR10,
        train_kwargs={'train': True},
        eval_train_kwargs={'train': True},
        eval_test_kwargs={'train': False},
        target_attr='targets',
    ),
    'cifar100': DatasetMetadata(
        name='cifar100',
        image_size=32,
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
        num_classes=100,
        dataset_cls=CIFAR100,
        train_kwargs={'train': True},
        eval_train_kwargs={'train': True},
        eval_test_kwargs={'train': False},
        target_attr='targets',
    ),
    'stl10': DatasetMetadata(
        name='stl10',
        image_size=96,
        mean=(0.4467, 0.4398, 0.4066),
        std=(0.2603, 0.2566, 0.2713),
        num_classes=10,
        dataset_cls=STL10,
        train_kwargs={'split': 'train+unlabeled'},
        eval_train_kwargs={'split': 'train'},
        eval_test_kwargs={'split': 'test'},
        target_attr='labels',
    ),
}


class ContrastivePairDataset(Dataset):
    def __init__(self, base_dataset, transform, target_getter=None):
        self.base_dataset = base_dataset
        self.transform = transform
        self.target_getter = target_getter

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        img, target = self.base_dataset[index]

        pos_1 = self.transform(img) if self.transform is not None else img
        pos_2 = self.transform(img) if self.transform is not None else img

        if self.target_getter is not None:
            target = self.target_getter(index)

        return pos_1, pos_2, target

    def get_targets(self):
        if self.target_getter is None:
            raise ValueError('Target getter is required to extract dataset targets.')
        return self.target_getter()


class EvalDataset(Dataset):
    def __init__(self, base_dataset, transform, target_getter=None):
        self.base_dataset = base_dataset
        self.transform = transform
        self.target_getter = target_getter

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        img = self.transform(img) if self.transform is not None else img

        if self.target_getter is not None:
            target = self.target_getter(index)

        return img, target

    def get_targets(self):
        if self.target_getter is None:
            raise ValueError('Target getter is required to extract dataset targets.')
        return self.target_getter()


def get_dataset_metadata(name):
    try:
        return DATASET_REGISTRY[name.lower()]
    except KeyError as exc:
        supported = ', '.join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(f'Unsupported dataset "{name}". Expected one of: {supported}.') from exc


def build_train_transform(image_size, mean, std):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_eval_transform(image_size, mean, std):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_target_getter(dataset, target_attr):
    def getter(index=None):
        targets = getattr(dataset, target_attr)
        if isinstance(targets, torch.Tensor):
            values = targets.tolist()
        elif hasattr(targets, 'tolist'):
            values = targets.tolist()
        else:
            values = list(targets)

        if index is None:
            return values
        return int(values[index])

    return getter


def _build_base_dataset(metadata, root, dataset_kwargs, download=True):
    return metadata.dataset_cls(root=root, transform=None, download=download, **dataset_kwargs)


def build_pretrain_datasets(name, root, download=True):
    metadata = get_dataset_metadata(name)
    train_transform = build_train_transform(metadata.image_size, metadata.mean, metadata.std)
    eval_transform = build_eval_transform(metadata.image_size, metadata.mean, metadata.std)

    train_base = _build_base_dataset(metadata, root, metadata.train_kwargs, download=download)
    memory_base = _build_base_dataset(metadata, root, metadata.eval_train_kwargs, download=download)
    test_base = _build_base_dataset(metadata, root, metadata.eval_test_kwargs, download=download)

    return (
        ContrastivePairDataset(
            train_base,
            train_transform,
            target_getter=build_target_getter(train_base, metadata.target_attr),
        ),
        EvalDataset(
            memory_base,
            eval_transform,
            target_getter=build_target_getter(memory_base, metadata.target_attr),
        ),
        EvalDataset(
            test_base,
            eval_transform,
            target_getter=build_target_getter(test_base, metadata.target_attr),
        ),
    )


def build_linear_eval_datasets(name, root, download=True):
    metadata = get_dataset_metadata(name)
    train_transform = build_train_transform(metadata.image_size, metadata.mean, metadata.std)
    eval_transform = build_eval_transform(metadata.image_size, metadata.mean, metadata.std)

    train_data = metadata.dataset_cls(
        root=root,
        transform=train_transform,
        download=download,
        **metadata.eval_train_kwargs,
    )
    test_data = metadata.dataset_cls(
        root=root,
        transform=eval_transform,
        download=download,
        **metadata.eval_test_kwargs,
    )
    return train_data, test_data
