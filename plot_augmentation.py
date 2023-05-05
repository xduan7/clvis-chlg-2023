"""This file aims to explore the augmentation of the dataset."""
import matplotlib.pyplot as plt
from torchvision import transforms

from benchmarks import get_cifar_based_benchmark

aug_tsfm = transforms.Compose(
    [
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
        ),
        # transforms.RandomRotation(
        #     degrees=15,
        #     expand=False,
        # )
    ]
)

benchmark = get_cifar_based_benchmark(
    scenario_config="config_s1.pkl",
    seed=0,
    benchmark=False,
)
dataset = benchmark.train_stream[0].dataset
dataset = dataset.remove_current_transform_group()

cifar100_classes = [
    "beaver",
    "dolphin",
    "otter",
    "seal",
    "whale",
    "aquarium fish",
    "flatfish",
    "ray",
    "shark",
    "trout",
    "orchids",
    "poppies",
    "roses",
    "sunflowers",
    "tulips",
    "bottles",
    "bowls",
    "cans",
    "cups",
    "plates",
    "apples",
    "mushrooms",
    "oranges",
    "pears",
    "sweet peppers",
    "clock",
    "computer keyboard",
    "lamp",
    "telephone",
    "television",
    "bed",
    "chair",
    "couch",
    "table",
    "wardrobe",
    "bee",
    "beetle",
    "butterfly",
    "caterpillar",
    "cockroach",
    "bear",
    "leopard",
    "lion",
    "tiger",
    "wolf",
    "bridge",
    "castle",
    "house",
    "road",
    "skyscraper",
    "cloud",
    "forest",
    "mountain",
    "plain",
    "sea",
    "camel",
    "cattle",
    "chimpanzee",
    "elephant",
    "kangaroo",
    "fox",
    "porcupine",
    "possum",
    "raccoon",
    "skunk",
    "crab",
    "lobster",
    "snail",
    "spider",
    "worm",
    "baby",
    "boy",
    "girl",
    "man",
    "woman" "crocodile",
    "dinosaur",
    "lizard",
    "snake",
    "turtle",
    "hamster",
    "mouse",
    "rabbit",
    "shrew",
    "squirrel",
    "maple",
    "oak",
    "palm",
    "pine",
    "willow",
    "bicycle",
    "bus",
    "motorcycle",
    "pickup truck",
    "train",
    "lawn-mower",
    "rocket",
    "streetcar",
    "tank",
    "tractor",
]


def plot_augmented_images(index):
    image = dataset[index][0]
    label = dataset[index][1]
    # Plot the original image and the 16 augmented images with matplotlib
    fig, axs = plt.subplots(8, 8, figsize=(16, 16))
    axs[0, 0].imshow(image)
    axs[0, 0].axis("off")
    for i in range(8):
        for j in range(8):
            if i == 0 and j == 0:
                continue
            axs[i, j].imshow(aug_tsfm(image))
            axs[i, j].axis("off")
    plt.suptitle(f"Label: {cifar100_classes[label]} (not)")
    plt.show()


# plot_augmented_images(1)
