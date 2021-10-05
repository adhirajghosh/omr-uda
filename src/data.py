import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
import numpy as np
import scipy.io
import gzip
#import wget
import h5py
import pickle
import urllib
import os
import skimage
import skimage.transform
from skimage.io import imread
import matplotlib.image as mpimg
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from PIL import Image


def train_val_dataset(dataset, test_split=0.25):
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['test'] = Subset(dataset, test_idx)
    return datasets

def LoadDataset(name, root, batch_size, split,shuffle=True, style=None, transform = None):
    if name == 'mnist':
        if split == 'train':
            return LoadMNIST(root+'mnist/', batch_size=batch_size, split='train', shuffle=shuffle, scale_32=True)
        elif split=='test':
            return LoadMNIST(root+'mnist/', batch_size=batch_size, split='test', shuffle=False, scale_32=True)
    elif name == 'usps':
        if split == 'train':
            return LoadUSPS(root+'usps/', batch_size=batch_size, split='train', shuffle=shuffle, scale_32=True)
        elif split=='test':
            return LoadUSPS(root+'usps/', batch_size=batch_size, split='test', shuffle=False, scale_32=True)
    elif name == 'svhn':
        if split == 'train':
            return LoadSVHN(root+'svhn/', batch_size=batch_size, split='extra', shuffle=shuffle)
        elif split=='test':
            return LoadSVHN(root+'svhn/', batch_size=batch_size, split='test', shuffle=False)
    elif name == 'face':
        assert style != None
        if split == 'train':
            return LoadFace(root, style=style, split='train', batch_size=batch_size,  shuffle=shuffle)
        elif split=='test':
            return LoadFace(root, style=style, split='test', batch_size=batch_size,  shuffle=False)
    elif name == 'imslp':
        if split == 'train':
            return LoadIMSLP(root+'imslp_dataset/', transform, batch_size=batch_size, data_split = 'train', shuffle=True)
        elif split=='test':
            return LoadIMSLP(root+'imslp_dataset/', transform, batch_size=batch_size, data_split='test', shuffle=False)

    elif name == 'realscore':
        if split == 'train':
            return LoadRealScore(root+'/ds2_dense/', transform, batch_size=batch_size, data_split = 'train', shuffle=True)
        elif split=='test':
            return LoadRealScore(root+'/ds2_dense/', transform, batch_size=batch_size, data_split='test', shuffle=False)

def LoadIMSLP(data_root, transform, batch_size=32, data_split='train', shuffle=True):
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    imslp_dataset = Imslp(root_dir=data_root, split=data_split, transform = transform)
    return DataLoader(imslp_dataset,batch_size=batch_size, shuffle=shuffle, drop_last=True)

def LoadRealScore(data_root, transform, batch_size=32, data_split='train', shuffle=True):
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    realscore_dataset = RealScore(root_dir=data_root, split=data_split, transform = transform)
    return DataLoader(realscore_dataset,batch_size=batch_size, shuffle=shuffle, drop_last=True)


def LoadSVHN(data_root, batch_size=32, split='train', shuffle=True):
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    svhn_dataset = datasets.SVHN(data_root, split=split, download=True,
                                   transform=transforms.ToTensor())
    return DataLoader(svhn_dataset,batch_size=batch_size, shuffle=shuffle, drop_last=True)

def LoadUSPS(data_root, batch_size=32, split='train', shuffle=True, scale_32 = False):
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    usps_dataset = USPS(root=data_root,train=(split=='train'),download=True,scale_32=scale_32)
    return DataLoader(usps_dataset,batch_size=batch_size, shuffle=shuffle, drop_last=True)

def LoadMNIST(data_root, batch_size=32, split='train', shuffle=True, scale_32 = False):
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    if scale_32:
        trans = transforms.Compose([transforms.Resize(size=[32, 32]),transforms.ToTensor()])
    else:
        trans = transforms.ToTensor()

    mnist_dataset = datasets.MNIST(data_root, train=(split=='train'), download=True,
                                   transform=trans)
    return DataLoader(mnist_dataset,batch_size=batch_size,shuffle=shuffle, drop_last=True)


def LoadFace(data_root, batch_size=32, split='train', style='photo', attr = None,
               shuffle=True, load_first_n = None):

    data_root = data_root+'face.h5'
    key = '/'.join(['CelebA',split,style])
    celeba_dataset = Face(data_root,key,load_first_n)
    return DataLoader(celeba_dataset,batch_size=batch_size,shuffle=shuffle,drop_last=True)


class Imslp(Dataset):
    """
    The Class will act as the container for our dataset. It will take your dataframe, the root path, and also the transform function for transforming the dataset.
    """

    def __init__(self, root_dir='', split='train', transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.dataset = self.process_dir()
        self.data_all = self.train_val_dataset(test_split = 0.25)
        self.train = self.data_all['train']
        print(len(self.train))
        self.test = self.data_all['test']
        print(len(self.test))
        self.split = split

    def __len__(self):
        # Return the length of the dataset
        if self.split == 'train':
            return len(self.train)
        else:
            return len(self.test)

    def __getitem__(self, idx):
        # Return the observation based on an index. Ex. dataset[0] will return the first element from the dataset, in this case the image and the label.
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.split == 'train':
            img_name = self.train[idx]
        else:
            img_name = self.test[idx]
        #img_name = self.data[idx]
        image = Image.open(img_name)
        #label = self.data_frame.iloc[idx, -1]

        image = self.transform(image)
        return (image)

    def process_dir(self):
        dataset = []
        for i in sorted(os.listdir(self.root_dir)):
            for j in sorted(os.listdir(os.path.join(self.root_dir, i))):
                dataset.append(os.path.join(self.root_dir, i, j))
        return dataset

    def train_val_dataset(self, test_split=0.25):
        train_idx, test_idx = train_test_split(list(range(len(self.dataset))), test_size=test_split)
        datasets = {}
        datasets['train'] = Subset(self.dataset, train_idx)
        datasets['test'] = Subset(self.dataset, test_idx)
        return datasets


class RealScore(Dataset):
    """
    The Class will act as the container for our dataset. It will take your dataframe, the root path, and also the transform function for transforming the dataset.
    """

    def __init__(self, root_dir='', split='train', transform=None):

        self.root_dir = root_dir+'/images/'
        self.transform = transform
        self.dataset = self.process_dir()
        self.data_all = self.train_val_dataset(test_split = 0.25)
        self.train = self.data_all['train']
        print(len(self.train))
        self.test = self.data_all['test']
        print(len(self.test))
        self.split = split

    def __len__(self):
        # Return the length of the dataset
        if self.split == 'train':
            return len(self.train)
        else:
            return len(self.test)

    def __getitem__(self, idx):
        # Return the observation based on an index. Ex. dataset[0] will return the first element from the dataset, in this case the image and the label.
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.split == 'train':
            img_name = self.train[idx]
        else:
            img_name = self.test[idx]

        #img_name = self.data[idx]
        image = Image.open(img_name)
        #label = self.data_frame.iloc[idx, -1]

        image = self.transform(image)
        return (image)

    def process_dir(self):
        dataset = []
        for i in sorted(os.listdir(self.root_dir)):
            dataset.append(os.path.join(self.root_dir, i))
        return dataset

    def train_val_dataset(self, test_split=0.25):
        train_idx, test_idx = train_test_split(list(range(len(self.dataset))), test_size=test_split)
        datasets = {}
        datasets['train'] = Subset(self.dataset, train_idx)
        datasets['test'] = Subset(self.dataset, test_idx)
        return datasets

### USPS Reference : https://github.com/corenel/torchzoo/blob/master/torchzoo/datasets/usps.py
class USPS(Dataset):
    """USPS Dataset.
    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN_PyTorch/master/data/uspssample/usps_28x28.pkl"

    def __init__(self, root, train=True, scale_32=False, download=False):
        """Init USPS dataset."""
        # init params
        self.root = os.path.expanduser(root)

        if scale_32:
            self.filename = "usps_32x32.pkl"
        else:
            self.filename = "usps_28x28.pkl"
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.dataset_size = None

        # download dataset.
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        self.train_data, self.train_labels = self.load_samples()
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], ::]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]

        #self.train_data *= 255.0
        #self.train_data = self.train_data.transpose(
        #    (0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.train_data[index, ::], self.train_labels[index]
        label = torch.LongTensor([np.int64(label).item()])
        # label = torch.FloatTensor([label.item()])
        return torch.FloatTensor(img), label[0]

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        """Download dataset."""
        filename = os.path.join(self.root, 'usps_28x28.pkl')
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if not os.path.isfile(filename):
            print("Download %s to %s" % (self.url, os.path.abspath(filename)))
            #urllib.request.urlretrieve(self.url, filename)
            wget.download(self.url,out=os.path.join(self.root, 'usps_28x28.pkl'))
            print("[DONE]")
        if not os.path.isfile(os.path.join(self.root, 'usps_32x32.pkl')):
            print("Resizing USPS 28x28 to 32x32...")
            f = gzip.open(os.path.join(self.root, 'usps_28x28.pkl'), "rb")
            data_set = pickle.load(f, encoding="bytes")
            for d in [0,1]:
                tmp = []
                for img in range(data_set[d][0].shape[0]):
                    tmp.append(np.expand_dims(skimage.transform.resize(data_set[d][0][img].squeeze(),[32,32]),0))
                data_set[d][0] = np.array(tmp)
            fp=gzip.open(os.path.join(self.root, 'usps_32x32.pkl'),'wb')
            pickle.dump(data_set,fp)
            print("[DONE")
        return

    def load_samples(self):
        """Load sample images from dataset."""
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels

class Face(Dataset):
    def __init__(self, root, key, load_first_n = None):

        with h5py.File(root,'r') as f:
            data = f[key][()]
            if load_first_n:
                data = data[:load_first_n]
        self.imgs = (data/255.0)*2 -1

    def __getitem__(self, index):
        return self.imgs[index]

    def __len__(self):
        return len(self.imgs)