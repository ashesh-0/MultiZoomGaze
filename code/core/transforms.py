import torchvision.transforms as transforms


def centercrop_transform(crop_size, img_size):
    image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(crop_size),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        image_normalize,
    ])
