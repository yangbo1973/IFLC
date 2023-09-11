from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.io import read_image
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter



class animal_dataset(Dataset): 
    def __init__(self, dataset, root_dir, transform, divide_mode, pred=[], probability=[], log = ''): 

        self.transform = transform
        self.divide_mode = divide_mode  
        
        if self.divide_mode=='test':
            dir_ = os.path.join(root_dir, 'testing')
            for root, dirs, files in os.walk(dir_):
                pass
            self.files = files
            test_data = []
            for file in self.files:
                image = read_image(os.path.join(dir_, file))
                data = np.array(image)
                data = np.expand_dims(data, 0)
                test_data.append(data)
            
            test_data = np.concatenate(test_data, axis = 0)
            test_data = test_data.reshape((5000, 3, 64, 64))
            test_data = test_data.transpose((0, 2, 3, 1))      
            test_label = np.array([int(f.split('_')[0]) for f in files])
            self.test_data = test_data
            self.test_label = test_label
            np.save(os.path.join(root_dir,'test_Y.npy'), self.test_label)
                               
        else:    
            dir_ = os.path.join(root_dir, 'training')
            for root, dirs, files in os.walk(dir_):
                pass
            self.files = files
            train_data = []
            for file in self.files:
                image = read_image(os.path.join(dir_, file))
                data = np.array(image)
                data = np.expand_dims(data, 0)
                train_data.append(data)
            
            train_data = np.concatenate(train_data, axis = 0)
            train_data = train_data.reshape((50000, 3, 64, 64))
            train_data = train_data.transpose((0, 2, 3, 1))      
            train_label = np.array([int(f.split('_')[0]) for f in files])
            noise_label = train_label
            np.save(os.path.join(root_dir,'train_Y.npy'), train_label)

        if self.divide_mode == 'all':
            self.train_data = train_data
            self.noise_label = noise_label
        elif self.divide_mode in ["labeled", "unlabeled"]:                   
            if self.divide_mode == "labeled":
                pred_idx = pred.nonzero()[0]
                self.probability = [probability[i] for i in pred_idx]   

            elif self.divide_mode == "unlabeled":
                pred_idx = (1-pred).nonzero()[0]                                               

            self.train_data = train_data[pred_idx]
            self.noise_label = [noise_label[i] for i in pred_idx]    
            self.predidx = pred_idx
            print("%s data has a size of %d"%(self.divide_mode,len(self.noise_label)))   
        elif self.divide_mode == 'test':
            pass
        else:            
            raise "wrong divide_mode %s"%self.divide_mode

            
    def __getitem__(self, index):
        if self.divide_mode=='labeled':
            img, target, probability = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, probability, index      
        elif self.divide_mode=='unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, index
        elif self.divide_mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index          
        
        elif self.divide_mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index
           
    def __len__(self):
        if self.divide_mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
        
class animal_dataloader():  
    def __init__(self, dataset, batch_size, num_workers, root_dir, log,):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        
        self.transform_train = transforms.Compose([
                transforms.Resize(112),
                transforms.RandomResizedCrop(112, scale = (0.3,1)),
                transforms.RandomHorizontalFlip(0.5),    
                transforms.ToTensor(),
                transforms.Normalize((131.01/255, 123.50/255, 107.43/255),(69.11/255, 67.92/255,70.62/255)),
            ]) 
        self.transform_test = transforms.Compose([
                transforms.Resize(112),
                transforms.ToTensor(),
                transforms.Normalize((131.01/255, 123.50/255, 107.43/255),(69.11/255, 67.92/255,70.62/255)),
            ])    
            
    def run(self,train_mode,pred=[],probability=[]):
        if train_mode=='warmup':
            all_dataset = animal_dataset(dataset=self.dataset, root_dir=self.root_dir, transform=self.transform_train, divide_mode="all",)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return all_dataset, trainloader
                                     
        elif train_mode=='train':
            labeled_dataset = animal_dataset(dataset=self.dataset, root_dir=self.root_dir, transform=self.transform_train, divide_mode="all",)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            return labeled_dataset, labeled_trainloader
        
        elif train_mode=='train_SSL':
            labeled_dataset = animal_dataset(dataset=self.dataset,
                                            divide_mode = 'labeled',
                                            root_dir = self.root_dir,
                                            transform = self.transform_train,
                                            pred = pred, probability = probability,log = self.log
                                            )              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            unlabeled_dataset = animal_dataset(dataset=self.dataset,
                                            divide_mode = 'unlabeled',
                                            root_dir = self.root_dir,
                                            transform = self.transform_train,
                                            pred = pred, probability = probability,
                                            )              
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            return labeled_dataset, labeled_trainloader, unlabeled_dataset, unlabeled_trainloader
        
        elif train_mode=='test':
            test_dataset = animal_dataset(dataset=self.dataset, root_dir=self.root_dir, transform=self.transform_test, divide_mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_dataset, test_loader
        
        elif train_mode=='eval_train':
            eval_dataset = animal_dataset(dataset=self.dataset, root_dir=self.root_dir, transform=self.transform_test, divide_mode='all',)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_dataset, eval_loader        