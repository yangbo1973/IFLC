from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter
from Asymmetric_Noise import *

            
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset): 
    def __init__(self, dataset, noise_mode, root_dir, transform, divide_mode, noise_file='', pred=[], probability=[], log = ''): 

        real_world_noise_types = ['aggre','worst','rand1','rand2','rand3','noisy100']
        synthetic_noise_types = ['sym','asym']
        
        if noise_mode.split('_')[0] in synthetic_noise_types:
            r = float(noise_mode.split('_')[1])
            noise_mode = noise_mode.split('_')[0]
        elif noise_mode in real_world_noise_types:
            r = 0.
        else:
            raise "wrong noise_mode %s"%noise_mode
                    
        self.r = r # noise ratio
        self.noise_mode = noise_mode
        self.transform = transform
        self.divide_mode = divide_mode  
        self.noise_file = noise_file
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
        
        if self.divide_mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']                            
        else:    
            train_data=[]
            train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset=='cifar100':
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            self.train_label = train_label
  
            if os.path.exists(noise_file) and noise_mode in ["sym","asym"]:
                noise_label = json.load(open(noise_file,"r"))
            else:

                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.r*50000)            
                noise_idx = idx[:num_noise]

                if noise_mode in real_world_noise_types:
                    noise_type_map = {'worst': 'worse_label',
                                      'aggre': 'aggre_label',
                                      'rand1': 'random_label1',
                                      'rand2': 'random_label2',
                                      'rand3': 'random_label3',
                                      'clean100': 'clean_label',
                                      'noisy100': 'noisy_label'}
                    noise_mode = noise_type_map[noise_mode]
                    self.noise_mode = noise_mode
                    noise_label = self.load_label().tolist()

                else:
                    if noise_mode == 'asym':
                        if dataset== 'cifar100':
                            noise_label, prob11 =  noisify_cifar100_asymmetric(train_label, self.r)
                            noise_label = noise_label.tolist()
                        else:
                            for i in range(50000):
                                if i in noise_idx:
                                        noiselabel = self.transition[train_label[i]]
                                        noise_label.append(noiselabel)
                                else:
                                    noise_label.append(train_label[i])   
                    elif noise_mode == "sym":
                        for i in range(50000):
                            if i in noise_idx:
                                # if noise_mode=='sym':
                                if dataset=='cifar10': 
                                    noiselabel = random.randint(0,9)
                                elif dataset=='cifar100':    
                                    noiselabel = random.randint(0,99)
                                noise_label.append(noiselabel)

                                # elif noise_mode=='pair_flip':  
                                #     noiselabel = self.pair_flipping[train_label[i]]
                                #     noise_label.append(noiselabel)   

                            else:
                                noise_label.append(train_label[i])  
                    else:
                        raise "wrong noise_mode %s"%noise_mode
                    print("save noisy labels to %s ..."%noise_file)        
                    json.dump(noise_label,open(noise_file,"w"))      
            
            if self.divide_mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            elif self.divide_mode in ["labeled", "unlabeled"]:                   
                if self.divide_mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    
                    clean = (np.array(noise_label)==np.array(train_label))                                                       
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability,clean)        
                    auc,_,_ = auc_meter.value()               
                    log.write('Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                    log.flush()      
                    
                elif self.divide_mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                
                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]    
                self.predidx = pred_idx
                print("%s data has a size of %d"%(self.divide_mode,len(self.noise_label)))   
            else:
                raise "wrong divide_mode %s"%self.divide_mode
                
    def load_label(self):
        #NOTE only load manual training label
        noise_label = torch.load(self.noise_file)
        if isinstance(noise_label, dict):
            if "clean_label" in noise_label.keys():
                clean_label = torch.tensor(noise_label['clean_label'])
                assert torch.sum(torch.tensor(self.train_label) - clean_label) == 0  
                print(f'Loaded {self.noise_mode} from {self.noise_file}.')
                print(f'The overall noise rate is {1-np.mean(clean_label.numpy() == noise_label[self.noise_mode])}')
            return noise_label[self.noise_mode].reshape(-1)  
        else:
            raise Exception('Input Error')
            
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
        
        elif self.divide_mode=='clean_train':
            img, target = self.train_data[index], self.clean_label[index]
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
        
        
class cifar_dataloader():  
    def __init__(self, dataset, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        if self.dataset=='cifar10':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])    
        elif self.dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])   
            
    def run(self,train_mode,pred=[],probability=[]):
        if train_mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, root_dir=self.root_dir, transform=self.transform_train, divide_mode="all",noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return all_dataset, trainloader
                                     
        elif train_mode=='train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, root_dir=self.root_dir, transform=self.transform_train, divide_mode="all", noise_file=self.noise_file,)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            return labeled_dataset, labeled_trainloader
        
        elif train_mode=='train_SSL':
            labeled_dataset = cifar_dataset(dataset=self.dataset,
                                            divide_mode = 'labeled',
                                            root_dir = self.root_dir,
                                            transform = self.transform_train,
                                            noise_mode = self.noise_mode,
                                            noise_file = self.noise_file,
                                            pred = pred, probability = probability,log = self.log
                                            )              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            unlabeled_dataset = cifar_dataset(dataset=self.dataset,
                                            divide_mode = 'unlabeled',
                                            root_dir = self.root_dir,
                                            transform = self.transform_train,
                                            noise_mode = self.noise_mode,
                                            noise_file = self.noise_file,
                                            pred = pred, probability = probability,
                                            )              
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            return labeled_dataset, labeled_trainloader, unlabeled_dataset, unlabeled_trainloader
        
        elif train_mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, root_dir=self.root_dir, transform=self.transform_test, divide_mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_dataset, test_loader
        
        elif train_mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, root_dir=self.root_dir, transform=self.transform_test, divide_mode='all', noise_file=self.noise_file)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_dataset, eval_loader        