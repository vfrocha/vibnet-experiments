from torchvision.transforms import v2

def get_train_augmentation():
    transform = v2.Compose([v2.ToTensor(),
                            v2.Resize((512,512)),
                            v2.RandomHorizontalFlip(p=0.2),
                            v2.RandomVerticalFlip(p=0.2),
                            v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    
    return transform

def get_test_augmentation():
    transform = v2.Compose([v2.ToTensor(),
                            v2.Resize((512,512)),
                            v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    return transform