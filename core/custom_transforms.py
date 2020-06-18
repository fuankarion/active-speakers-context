import os
from torchvision import transforms

video_train = transforms.Compose([
    transforms.Resize((144, 144)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize((0.2324, 0.2721, 0.3448), (0.1987, 0.2172, 0.2403))
])

video_val = transforms.Compose([
    transforms.Resize((144, 144)),
    transforms.ToTensor(),
    transforms.Normalize((0.2324, 0.2721, 0.3448), (0.1987, 0.2172, 0.2403))
])
