from torchvision.transforms import Compose, RandomHorizontalFlip, ColorJitter, RandomRotation, RandomAffine, RandomPerspective, ToTensor, Normalize

transform = Compose([
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.3, contrast=0.3),
    RandomRotation(20),
    RandomAffine(degrees=15),
    RandomPerspective(distortion_scale=0.2, p=0.5),
    ToTensor(),
    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
