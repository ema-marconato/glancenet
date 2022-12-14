import os
from .utils import Accuracy_Loss, F1_Loss, net

from kmodes.kmodes import KModes
import pickle

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import logging

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.optim import SGD

from common import constants as c

def validate_model(data,labels):
    print('Calculating whether the model is linearly separable')
    
    model = net(40,10)
    
    optimizer = SGD(model.parameters(), 0.1, 0.9)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.cat((data, labels.view(-1,1)), dim=1)
    dataloader = DataLoader(dataset, batch_size=125)

    for epoch in range(51):
        
        for iterum, alls in enumerate(dataloader):
            X = alls[:,:40]
            y = alls[:,40:].to(torch.long)
            optimizer.zero_grad()
                    
            pred = model(X)
            loss = criterion(pred,y.view(-1))

            loss.backward()
            optimizer.step()
            
        if (epoch % 10)==0 :
            print('## Epoch:', epoch,'-> Loss:', loss)

    ## CALCULATE ACCURACY

    acc = 0
    bs = dataloader.batch_size
    tot = 0
    for data in dataloader:
        tot += 1
        X = data[:,:40]
        y = data[:,40:].view(-1).to(torch.long)
        classes = model.predict(X).to(torch.long)
        acc += (classes == y).sum()/bs      
    #acc = acc/tot

    print('Overall accuracy on train:', acc/tot)

def target_cast(labels, inputs, r_plane, irrelevant_components=[0], noise_fact=0.1, _plot=False):
    l = len(labels)

    targets = []
    admitted = np.ones(len(r_plane[0]))
    admitted[irrelevant_components] = 0
    print('Considered components', [iter for iter, i in enumerate(admitted) if i == 1])
    for i in range(len(inputs)):
        guess = (np.dot(inputs[i] - r_plane[1] * admitted, (r_plane[0] * admitted)))
        if guess < 0:
            target = 1
        else:
            target = 0
        if np.random.uniform() < noise_fact:
            target = (target + 1) % 2

        targets.append((target))
    for i in irrelevant_components:
        labels.remove(i)

    return np.array(targets, dtype=np.int)


def random_plane(labels, space, _plot=False):
    l = len(labels)

    random_versor = np.random.uniform(size=l)
    random_versor /= np.linalg.norm(random_versor)

    mean_vect = np.mean(space, axis=0)

    return [random_versor, mean_vect]


class LabelHandler(object):
    def __init__(self, labels, label_weights, class_values):
        self.labels = labels
        self._label_weights = None
        self._num_classes_torch = torch.tensor((0,))
        self._num_classes_list = [0]
        self._class_values = None
        if labels is not None:
            self._label_weights = [torch.tensor(w) for w in label_weights]
            self._num_classes_torch = torch.tensor([len(cv) for cv in class_values])
            self._num_classes_list = [len(cv) for cv in class_values]
            self._class_values = class_values

    def label_weights(self, i):
        return self._label_weights[i]

    def num_classes(self, as_tensor=True):
        if as_tensor:
            return self._num_classes_torch
        else:
            return self._num_classes_list

    def class_values(self):
        return self._class_values

    def get_label(self, idx):
        if self.labels is not None:
            return torch.tensor(self.labels[idx], dtype=torch.long)
        return None

    def get_values(self, idx):
        if self.labels is not None:
            return torch.tensor(self.labels[idx], dtype=torch.float)
        return None

    def has_labels(self):
        return self.labels is not None


class CustomImageFolder(ImageFolder):
    def __init__(self, root, labels, transform, name, num_channels, seed):
        super(CustomImageFolder, self).__init__(root, transform)
        self.indices = range(len(self))
        self._num_channels = num_channels
        self._name = name
        self.seed = seed

        #self.label_handler = LabelHandler(labels, label_weights, class_values)

        ## ADDED GRAYBOX VARIANT
        self.isGRAY = False

    @property
    def name(self):
        return self._name

    def label_weights(self, i):
        return self.label_handler.label_weights(i)

    def num_classes(self, as_tensor=True):
        return self.label_handler.num_classes(as_tensor)

    def class_values(self):
        return self.label_handler.class_values()

    def has_labels(self):
        return self.label_handler.has_labels()

    def num_channels(self):
        return self._num_channels

    def __getitem__(self, index1):
        path1 = self.imgs[index1][0]
        img1 = self.loader(path1)
        if self.transform is not None:
            img1 = self.transform(img1)

        if self.isGRAY:
            return img1
        
        label1 = 0
        if self.label_handler.has_labels():
            label1 = self.label_handler.get_label(index1)
        if self.isGRAY: label1 = self.label_weights(index1)

        return img1, label1
         
        


class CustomNpzDataset(Dataset): ### MODIFIED HERE THE DATABASE TYPE FOR _GET_ITEM
    def __init__(self, data_images, transform, labels, label_weights, name, class_values, num_channels, seed, examples, y_target):
        assert len(examples) == len(y_target), str(len(examples))+' '+str(len(y_target))

        self.seed = seed
        self.data_npz = data_images
        self._name = name
        self._num_channels = num_channels

        self.label_handler = LabelHandler(labels, label_weights, class_values)

        self.transform = transform
        self.indices = range(len(self))

        ## ADDED GRAYBOX VARIANT
        self.isGRAY = False
        self.y_data = y_target
        self.examples = examples
        
    @property
    def name(self):
        return self._name

    def label_weights(self, i):
        return self.label_handler.label_weights(i)

    def num_classes(self, as_tensor=True):
        return self.label_handler.num_classes(as_tensor)

    def class_values(self):
        return self.label_handler.class_values()

    def has_labels(self):
        return self.label_handler.has_labels()

    def num_channels(self):
        return self._num_channels

    def __getitem__(self, index1):
        #print("Passed index", index1)
        img1 = Image.fromarray(self.data_npz[index1])# * 255)
        if self.transform is not None:
            img1 = self.transform(img1)

        label1 = 0
        if self.label_handler.has_labels():
            label1 = self.label_handler.get_label(index1)
            #print("The obtained label is", label1)
            ### INSERTED THE TRUTH VALUE FOR DSPIRTES
            if self.isGRAY:
                y = self.y_data[index1]
                z_values = self.label_handler.get_values(index1)

                return img1, z_values, y, self.examples[index1]

        return img1, label1, 0, 0 # y, self.examples[index1]

    def __len__(self):
        return self.data_npz.shape[0]


class DisentanglementLibDataset(Dataset):
    """
    Data-loading from Disentanglement Library

    Note:
        Unlike a traditional Pytorch dataset, indexing with _any_ index fetches a random batch.
        What this means is dataset[0] != dataset[0]. Also, you'll need to specify the size
        of the dataset, which defines the length of one training epoch.

        This is done to ensure compatibility with disentanglement_lib.
    """

    def __init__(self, name, seed=0, make_yset=False):
        """
        Parameters
        ----------
        name : str
            Name of the dataset use. You may use `get_dataset_name`.
        seed : int
            Random seed.
        iterator_len : int
            Length of the dataset. This defines the length of one training epoch.
        """
        from disentanglement_lib.data.ground_truth.named_data import get_named_ground_truth_data
        self.name = name
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.dataset = get_named_ground_truth_data(self.name)
        self.iterator_len = self.dataset.images.shape[0]

    @staticmethod
    def has_labels():
        return False


    def num_channels(self):
        return self.dataset.observation_shape[2]

    def __len__(self):
        return self.iterator_len

    def __getitem__(self, item):
        assert item < self.iterator_len
        output = self.dataset.sample_observations(1, random_state=self.random_state)[0]
        # Convert output to CHW from HWC
        return torch.from_numpy(np.moveaxis(output, 2, 0), ).type(torch.FloatTensor), 0


def _get_dataloader_with_labels(name, dset_dir, batch_size, seed, num_workers, image_size, include_labels, pin_memory,
                                shuffle, droplast, masking_fact=100, d_version="full", n_classes=10, noise_class=0):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), ])
    labels = None
    label_weights = None
    label_idx = None
    label_names = None
    class_values = None

    # check if labels are provided as indices or names
    if include_labels is not None:
        try:
            int(include_labels[0])
            label_idx = [int(s) for s in include_labels]
        except ValueError:
            label_names = include_labels
    logging.info('include_labels: {}'.format(include_labels))

    make_yset = False

    if name.lower() == 'celeba':
        print('Pre-Processing CelebA')
        root = os.path.join(dset_dir, 'celebA/celebA64-full-10.npz')

#        labels_file = os.path.join(root, 'list_attr_celeba.csv')

        transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(), 
                ])

        npz = np.load(root)

        labels = npz['Y']

        print('labels', np.shape(labels))
        
        # LOAD CLASSIFICATION
        if n_classes == 4: km_files = os.path.join(dset_dir, 'celebA/try_km.pickle')
        else: km_files = os.path.join(dset_dir, 'celebA/new_km.pickle')
        with open(km_files, 'rb') as f:
            km = pickle.load(f)

        targets = km.predict(labels)
        targets = np.asarray(targets, dtype=int)
        
        ## CHECK PERCS OF CLASSES
        tot = []
        for i in range(n_classes):
            y_mask = (targets==i)
            tot.append( len(targets[y_mask])/len(targets) )
        print('All classes percentages:', tot)

        train_data_kwargs = {'data_images': npz['X'][:102635],
                            'labels': labels[:102635],
                            'label_weights': labels[:102635],
                            'class_values': labels[:102635],
                            'num_channels': 3,
                            'y_target': targets[:102635],
                            }
        val_data_kwargs  = {'data_images': npz['X'][102635:115013],
                            'labels': labels[102635:115013],
                            'label_weights': labels[102635:115013],
                            'class_values': labels[102635:115013],
                            'num_channels': 3,
                            'y_target': targets[102635:115013],
                            }
        test_data_kwargs  = {'data_images': npz['X'][115013:],
                            'labels': labels[115013:],
                            'label_weights': labels[115013:],
                            'class_values': labels[115013:],
                            'num_channels': 3,
                            'y_target': targets[115013:],
                            }
        
        train_dset = CustomNpzDataset
        val_dset   = CustomNpzDataset
        test_dset  = CustomNpzDataset
              
    elif name.lower() == 'mpi3d_toy':
        print('Pre-Processing mpi3d_toy')
        root = os.path.join(dset_dir, 'mpi3d_toy/mpi3d_toy_full.npz')

#        labels_file = os.path.join(root, 'list_attr_celeba.csv')

        transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(), 
                ])

        if d_version == "full":
            root = os.path.join(dset_dir, 'mpi3d_toy/mpi3d_toy_full.npz')
        elif d_version == "smaller":
            root = os.path.join(dset_dir, 'mpi3d_toy/mpi3d_toy_small.npz')
        else:
            NotImplementedError('wrong choice of d_version')

        npz = np.load(root)
        
        labels = npz['Y']

        print('labels', np.shape(labels))
        
        # LOAD CLASSIFICATION
        km_files = os.path.join(dset_dir, 'mpi3d_toy/km.pickle')
        with open(km_files, 'rb') as f:
            km = pickle.load(f)


        targets = km.predict(labels[:,:-2])
        targets = np.asarray(targets, dtype=int)
        
        ## CHECK PERCS OF CLASSES
        tot = []
        for i in range(n_classes):
            y_mask = (targets==i)
            tot.append( len(targets[y_mask])/len(targets) )
        print('All classes percentages:', tot)

        #quit()

        if False:
            all_attrs , all_targets = torch.as_tensor(labels, dtype=torch.float), torch.as_tensor(targets, dtype=torch.float) 
            validate_model(all_attrs, all_targets)

        ## create labels for mpi3d
        a = np.zeros((len(labels),6))
        b = np.zeros((len(labels),6))
        c = np.zeros((len(labels),3))
        d = np.zeros((len(labels),3))
        a[np.arange(len(labels)), np.asarray(labels[:,0], dtype=np.int) ] = 1
        b[np.arange(len(labels)), np.asarray(labels[:,1], dtype=np.int) ] = 1
        #
        c[np.arange(len(labels)), np.asarray(labels[:,3], dtype=np.int) ] = 1
        d[np.arange(len(labels)), np.asarray(labels[:,4], dtype=np.int) ] = 1
        
        labels[:,5:] /= np.max(labels[:,5]) 
        
        new_labels = np.hstack((a,b,labels[:,2].reshape(-1,1),c,d))
        labels_one_hot = np.hstack((new_labels,  labels[:,5:]))
        
        print('Shape of latents for mpi3d:', np.shape(labels_one_hot))
                
        assert np.max(labels_one_hot) == 1, 'Error on max: '+str(np.max(labels_one_hot))
        assert np.min(labels) == 0, 'Error on min'
        
        data_kwargs = {'data_images': npz['X'],
                       'labels': labels_one_hot,
                       'label_weights': labels,
                       'class_values': labels,
                       'num_channels': 3,
                       'y_target': targets,
                       }
        dset = CustomNpzDataset
        
    elif name.lower() == 'dsprites_full':
        #print(name)
        if d_version == "full":
            root = os.path.join(dset_dir, 'dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        elif d_version=="smaller":
            root = os.path.join(dset_dir, 'dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64_smaller.npz')
        else:
            raise NotImplementedError('Dsprites, passed invalid argument.')

        npz = np.load(root)

        if label_idx is not None:
            #print("Passed label_idx:",label_idx)
            labels = (npz['latents_values'][:, label_idx])

            ### SHIFTING ALL VALUES OVER THEIR MEANS
            Mean = np.mean(labels, axis=0 )
            for j in label_idx:
                if j > 1:
                    labels[:, j] -= np.min(labels[:,j])
                    labels[:, j] /= (np.max(labels[:, j] ) )
            if 1 in label_idx:
                index_shape = label_idx.index(1)
                labels[:, 1] -= 1

            ### CREATE THE H-STACK with the usual np onehot

            b = np.zeros((len(labels),3))
            b[np.arange(len(labels)), np.asarray(labels[:,1], dtype=np.int) ] = 1

            new_labels = np.hstack( (labels[:,0].reshape(-1,1) , b))
            labels_one_hot = np.hstack((new_labels, labels[:,2:] ) )

            # dsprite has uniformly distributed labels
            num_labels = labels.shape[1]
            label_weights = []
            class_values = []
            for i in range(num_labels):
                unique_values, count = np.unique(labels[:, i], axis=0, return_counts=True)
                weight = 1 - count / labels.shape[0]
                if len(weight) == 1:
                    weight = np.array(1)
                else:
                    weight /= sum(weight)
                label_weights.append(np.array(weight))

                # always set label values to integers starting from zero
                unique_values_mock = np.arange(len(unique_values))
                class_values.append(unique_values_mock)
            label_weights = np.array(label_weights)#, dtype=np.float32)

        data_kwargs = {'data_images': npz['imgs']*255}
        data_kwargs.update({'labels': labels_one_hot,
                       'label_weights': label_weights,
                       'class_values': class_values,
                       'num_channels': 1})
        dset = CustomNpzDataset
        #print("The r plane:",[random_plane(label_idx, ranges)])
        if labels is not None:
            ### NOW CREATE A PARTICULAR TYPE THAT PREDICTS ONLY WITH TWO COMPONENTS
            r_plane=random_plane(label_idx, labels,  _plot=False)
            r_plane[0] = np.array([0,-0.55329954, -0.3754565, 0.46785691, 0.37617377, 0.4387428]) # JUST A RANDOM VECTOR
            target_set = np.asarray(target_cast(label_idx, labels, r_plane,
                                    irrelevant_components=[0], _plot=True, noise_fact=noise_class), dtype=np.int)
            print("target", np.shape(target_set))
            data_kwargs.update({'y_target': target_set})

            print("Population of 0:", len(target_set[target_set == 0]) / len(target_set) * 100, "%.")

            print("Start verification")
            in_data = labels

            print("Labels in data_loader")

            y_target1 = target_set

            lr = LogisticRegression( solver='lbfgs')#, penalty='none')
            lr.fit(in_data[:,1:], y_target1)
            y_pred = lr.predict_proba(in_data[:,1:])
 
 
            baseline2 = torch.nn.BCELoss(reduction='mean')(torch.tensor(y_pred[:,1], dtype=torch.float),
                                                           torch.tensor(y_target1, dtype=torch.float) )
 
            print("Fitting a LogReg model, loss - CE:",  baseline2)

            accuracy = Accuracy_Loss()((torch.tensor(y_pred, dtype=torch.float)), torch.tensor(y_target1, dtype=torch.float))
            print("ACCURACY for logreg: ", accuracy)

    elif name.lower() == 'dsprites_leakage':
        print('Pre-Processing dsprites_leakage')

        root = os.path.join(dset_dir, 'dsprites_leakage_train.npz')
        npz_train = np.load(root)
        
        labels = npz_train['G']
        targets = np.array(npz_train['Y']-1, dtype=int)

        print('labels', np.shape(labels))
                
        ## create labels for dsprites
        a = np.zeros((len(labels),3))
        a[np.arange(len(labels)), np.asarray(labels[:,0]-1, dtype=np.int) ] = 1
        
        for i in range(1,len(labels[0])):
            labels[:,i] = (labels[:,i] - np.min(labels[:,i])) / (np.max(labels[:,i]) - np.min(labels[:,i]) )
        
        labels_one_hot = np.hstack((a,  labels[:,1:]))
        
        print('Shape of latents for dsprites leakage:', np.shape(labels_one_hot))
                
        assert np.max(labels_one_hot) == 1, 'Error on max: '+str(np.max(labels_one_hot))
        assert np.min(labels) == 0, 'Error on min'
        
        # create a perm for train / val
        perm = np.random.permutation(len(labels_one_hot))

        validation_split = 0.2
        split = int(np.floor(validation_split * len(labels_one_hot) ))
        
        train_data_kwargs = {'data_images': npz_train['X'][perm][split:]*255 ,
                       'labels': labels_one_hot[perm][split:],
                       'label_weights': labels_one_hot[perm][split:],
                       'class_values': labels_one_hot[perm][split:],
                       'num_channels': 1,
                       'y_target': targets[perm][split:],
                       }
        
        val_data_kwargs  = {'data_images': npz_train['X'][perm][:split]*255,
                       'labels': labels_one_hot[perm][:split],
                       'label_weights': labels_one_hot[perm][:split],
                       'class_values': labels_one_hot[perm][:split],
                       'num_channels': 1,
                       'y_target': targets[perm][:split],
                       }
        
        root = os.path.join(dset_dir, 'dsprites_leakage_test.npz')
        npz_test = np.load(root)
        
        ## create labels for dsprites
        labels = npz_test['G']

        a = (labels[:,0] - 1).reshape(-1,1)
        
        for i in range(1,len(labels[0])):
            labels[:,i] = (labels[:,i] - np.min(labels[:,i])) / (np.max(labels[:,i]) - np.min(labels[:,i]) )
        
        labels_one_hot = np.hstack((a,  labels[:,1:]))
        
        test_data_kwargs  = {'data_images': npz_test['X']*255,
                            'labels': labels_one_hot,
                            'label_weights':labels_one_hot,
                            'class_values': labels_one_hot,
                            'num_channels': 1,
                            'y_target': npz_test['Y'],
                            }
        
        train_dset = CustomNpzDataset
        val_dset = CustomNpzDataset
        test_dset = CustomNpzDataset

        tot = [1/3, 1/3, 1/3]
    
    else:
        raise NotImplementedError

    ## PUT THE LABELED Z
    random_entries = np.random.uniform(0,1, size=(len(labels), ))
    examples = np.ones(len(labels))

    examples[~(random_entries <= (masking_fact/100) )] = 0
    if name.lower() not in ['celeba', 'dsprites_leakage']:
        data_kwargs.update({'seed': seed,
                            'name': name,
                            'transform': transform})
        

        data_kwargs.update({'examples': examples })

        dataset = dset(**data_kwargs)

        # Setting the Graybox here
        dataset.isGRAY = True

        # CREATING DATA LOADER + TEST LOADER

        validation_split = .2

        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        
        indices = list(range(dataset_size))
        np.random.seed(seed)
        np.random.shuffle(indices)
        split = int(np.floor(validation_split * dataset_size))
        train_indices, val_indices = indices[split:], indices[:split]

        l = len(val_indices)
        val_indices, test_indices = val_indices[:int(np.floor(l/2))], val_indices[int(np.floor(l/2)):]

        print('Dataset dimensions')
        print(len(train_indices), len(val_indices), len(test_indices) )
        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        data_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                drop_last=droplast,
                                sampler=train_sampler)

        val_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                drop_last=droplast,
                                sampler=val_sampler)

        test_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                drop_last=droplast,
                                sampler=test_sampler)
    else:
        ## CREATE 3 different datasets based on the train/val/test partition
        train_dset = CustomNpzDataset
        val_dset   = CustomNpzDataset
        test_dset  = CustomNpzDataset
        
        if name.lower() == 'celeba':
            train_examples = examples[:102635]
            val_examples   = examples[102635:115013]
            test_examples  = examples[115013:]
            
        elif name.lower() == 'dsprites_leakage':
            train_examples = np.ones(len(npz_train['Y'] ) - split  )
            val_examples   = np.ones(split)
            test_examples  = np.ones(len(npz_test['Y'])) 
        else:
            NotImplementedError('wrong dataset')
            
        train_data_kwargs.update({'seed': seed,
                                'name': name,
                                'transform': transform,
                                'examples': train_examples}) 
        
        train_dataset = train_dset(**train_data_kwargs)
        train_dataset.isGRAY =True 
        
        val_data_kwargs.update({'seed': seed,
                                'name': name,
                                'transform': transform,
                                'examples': val_examples}) 
        val_dataset = val_dset(**val_data_kwargs)
        val_dataset.isGRAY =True 
        
        test_data_kwargs.update({'seed': seed,
                                'name': name,
                                'transform': transform,
                                'examples': test_examples}) 
        test_dataset = test_dset(**test_data_kwargs)
        test_dataset.isGRAY =True 
                
        rel_weights = 1. / np.array(tot)
        weights = rel_weights[targets[:102635]]        
        
        #print(weights[:1000])
        
        weighted_sampler = WeightedRandomSampler(weights, 
                                                 len(weights), 
                                                 replacement=True)
        if name.lower() == 'celeba':
            data_loader = DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    drop_last=droplast,
                                    sampler=weighted_sampler
                                    )
            
        else:
            data_loader = DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    drop_last=droplast,
                                    )
            

        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                drop_last=droplast)

        test_loader = DataLoader(test_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                drop_last=droplast)
        
        #print(len(examples))
        #print(examples[:100])
      

    if include_labels is not None:
        logging.info('num_classes: {}'.format(dataset.num_classes(False)))
        logging.info('class_values: {}'.format(class_values))

    return data_loader, val_loader, test_loader



def get_dataset_name(name):
    """Returns the name of the dataset from its input argument (name) or the
    environment variable `AICROWD_DATASET_NAME`, in that order."""
    return name or os.getenv('AICROWD_DATASET_NAME', c.DEFAULT_DATASET)


def get_datasets_dir(dset_dir):
    if dset_dir:
        os.environ['DISENTANGLEMENT_LIB_DATA'] = dset_dir
    return dset_dir or os.getenv('DISENTANGLEMENT_LIB_DATA')


def _get_dataloader(name, batch_size, seed, num_workers, pin_memory, shuffle, droplast):
    """
    Makes a dataset using the disentanglement_lib.data.ground_truth functions, and returns a PyTorch dataloader.
    Image sizes are fixed to 64x64 in the disentanglement_lib.
    :param name: Name of the dataset use. Should match those of disentanglement_lib
    :return: DataLoader
    """
    dataset = DisentanglementLibDataset(name, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=droplast, pin_memory=pin_memory,
                        num_workers=num_workers,)
    return loader


def get_dataloader(dset_name, dset_dir, batch_size, seed, num_workers, image_size, include_labels, pin_memory,
                   shuffle, droplast, d_version="full", masking_fact=100,classes=10):
    locally_supported_datasets = c.DATASETS
    dset_name = get_dataset_name(dset_name)
    dsets_dir = get_datasets_dir(dset_dir)

    logging.info(f'Datasets root: {dset_dir}')
    logging.info(f'Dataset: {dset_name}')

    if dset_name in locally_supported_datasets:
        return _get_dataloader_with_labels(dset_name, dsets_dir, batch_size, seed, num_workers, image_size,
                                           include_labels, pin_memory, shuffle, droplast, d_version=d_version, masking_fact=masking_fact, 
                                           n_classes=classes)
    else:
        # use the dataloader of Google's disentanglement_lib
        return _get_dataloader(dset_name, batch_size, seed, num_workers, pin_memory, shuffle, droplast)

