from PIL import Image
import blobfile as bf
import mpi4py # ADDED
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the SECOND part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[1] for path in all_files] # CHANGE - CLASS IS SECOND PART OF FILENAME        
        # for CAMUS if label ('A2CED','A2CES','A4CED','A4CES') contains A2C or A4C replace with that 
        class_names = [val if 'A2C' not in val and 'A4C' not in val else 'A2C' if 'A2C' in val else 'A4C' for val in class_names]
        # create dict with label as key and index as value (starting at 0) to have labels 0...n
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))} # e.g. {'A2C': 0, 'A4C': 1}
        print('Number of classes:',len(sorted_classes))
        print('classes sorted:',sorted_classes)
        # convert list of labels to list of index (str2idx)
        classes = [sorted_classes[x] for x in class_names]
    # create dataset
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic: # if class conditioned is True
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        ) # num_workers=1 positive integer turns on multi-process data loading
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution # image size 112x112
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        
        # while image min dim is >= 2*desired img_size - rescale to box-shape (quadrant) with half size of min dim
        #while min(*pil_image.size) >= 2 * self.resolution:
            #pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
        
        # rescale image to desired image_size (resolution)
        #scale = self.resolution / min(*pil_image.size)
        #pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
        
        # if 1 colorchannel greyscale
        #if pil_image.mode == 'L':
        arr = np.array(pil_image) # convert to np array
        #print('ARR SHAPE',arr.shape)
        arr = arr.astype(np.float32) / 127.5 - 1 # normalize img from [0,255] to [-1,1]
        arr = np.expand_dims(arr, axis=2) # expand from (112,112) to (112,112,1)
        arr = np.transpose(arr, [2, 0, 1]) # flip shape from (112,112,1) to (1,112,112)
        
      
        # convert image to RGB and then numpy array
        #arr = np.array(pil_image.convert("RGB")) 
        # centre-crop by desired image_size (resolution)
        #crop_y = (arr.shape[0] - self.resolution) // 2
        #crop_x = (arr.shape[1] - self.resolution) // 2
        #arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        # normalizing img from [0,255] to [-1,1]
        #arr = arr.astype(np.float32) / 127.5 - 1
        #arr = np.transpose(arr, [2, 0, 1])

        # if class_cond = True then create dict with label as np array
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        
        return arr, out_dict
