import os
import numpy as np
import config
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        # Assume each first-level directory is a class
        self.class_names = sorted(os.listdir(root_dir))
        for index, class_name in enumerate(self.class_names):
            class_dir = os.path.join(root_dir, class_name)
            # Walk through all subdirectories
            for subdir, _, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # Save the full path to the image and its class index
                        self.data.append((os.path.join(subdir, file), index))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        image = np.array(Image.open(img_path))
        image = config.both_transforms(image=image)["image"]
        high_res = config.highres_transform(image=image)["image"]
        low_res = config.lowres_transform(image=image)["image"]
        return low_res, high_res



def test():
    dataset = MyImageFolder(root_dir="/content/flickr_face")
    loader = DataLoader(dataset, batch_size=1, num_workers=8)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)


if __name__ == "__main__":
    test()