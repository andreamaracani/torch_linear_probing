from multiprocessing import cpu_count

import torch

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer, SGD

from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms as T

from accelerate import Accelerator
from accelerate.data_loader import prepare_data_loader

from typing import Union, Tuple, Callable, Optional


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class FeatureExtractor(nn.Module):

    def __init__(self, 
                 model: nn.Module, 
                 get_features_fnc: Optional[Callable] = None) -> None:
        """
            Wrapper of a model to extract just features from its outputs.
        """
        super().__init__()
        self.model = model
        self.get_features = get_features_fnc

    def forward(self, x: torch.Tensor, args=None) -> torch.Tensor:

        if args is None:
            model_outs = self.model(x)
        else:
            model_outs = self.model(x, *args)

        if self.get_features is not None:
            features = self.get_features(model_outs)
        else:
            features = model_outs

        return features
    

class ExponentialScheduler(LambdaLR):

    def __init__(self, 
                 optimizer: Optimizer, 
                 max_steps: int, 
                 gamma: float = 10., 
                 power: float = 0.75) -> None:
        
        self.max_steps = max_steps
        self.gamma = gamma
        self.power = power
    
        decay_fnc = lambda step: (1 + gamma * step / max_steps) ** (-power)
        super().__init__(optimizer=optimizer, lr_lambda=decay_fnc)


def get_train_transform(mean: Tuple = IMAGENET_MEAN,
                        std: Tuple  = IMAGENET_STD,
                        resize_size: Union[int, Tuple[int, int]] = 256, 
                        crop_size:   Union[int, Tuple[int, int]] = 224) \
                            -> Callable:
    """ It returns a standard basic transform for training. """

    return  T.Compose([T.Resize(resize_size),
                       T.RandomCrop(crop_size),
                       T.RandomHorizontalFlip(),
                       T.ToTensor(),
                       T.Normalize(mean=mean, std=std)])


def get_test_transform(mean: Tuple = IMAGENET_MEAN,
                       std: Tuple  = IMAGENET_STD,
                       resize_size: Union[int, Tuple[int, int]] = 256, 
                       crop_size:   Union[int, Tuple[int, int]] = 224) \
                            -> Callable:
    """ It returns a standard basic transform for test. """

    return  T.Compose([T.Resize(resize_size),
                       T.CenterCrop(crop_size),
                       T.ToTensor(),
                       T.Normalize(mean=mean, std=std)])


@torch.inference_mode()
def extract_features(model: Union[FeatureExtractor, DDP],
                     dataset: ImageFolder,
                     batch_size: int, 
                     accelerator: Accelerator,
                     normalize: bool = True) -> dict:
    """
        Extracts and save the features from a model and the labels 

        Args:
            model(FeatureExtractor): a module that returns directly the features.
            dataset (ImageFolderIdx): an ImageFolder dataset.
            batch_size (int): the global batch size to use.
            accelerator (Accelerator): the accelerator.
            normalize (bool): True to normalize the features.
        Returns:
            the features and the labels.
    """
    


    # info from accelerator
    device = accelerator.device
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index

    loader = DataLoader(dataset=dataset, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        drop_last=False, 
                        num_workers=cpu_count())
    
    loader = prepare_data_loader(dataloader=loader, 
                                 device=device, 
                                 num_processes=num_processes, 
                                 process_index=process_index, 
                                 split_batches=True,
                                 even_batches=False, 
                                 put_on_device=True)

    model = model.eval()

    features_list = []
    labels_list   = []

    for images, labels in loader:
        features = model(images) 

        if normalize: features = F.normalize(features, dim=1)   

        features_list.append(accelerator.gather(features).cpu())
        labels_list.append(accelerator.gather(labels).cpu())
      
    features = torch.cat(features_list, dim=0)
    labels   = torch.cat(labels_list, dim=0)
    
    model = model.train()

    return features, labels 



class LinearClassifier:

    def __init__(self, 
                 in_features, 
                 n_classes,
                 accelerator: Accelerator,
                 use_bias: bool = True,
                 batch_size: int = 512,
                 epochs: int = 5,
                 lr: float = 1e-3,
                 label_smoothing: float = 0., 
                 gamma: float = 10., 
                 power: float = 0.75,
                 weight_decay: float = 1e-3,
                 momentum: float = 0.9,
                 nesterov: bool = True):


        self.n_classes   = n_classes
        self.accelerator = accelerator
        self.device = accelerator.device
        self.in_features = in_features
        self.classifier = nn.Linear(in_features, n_classes, bias=use_bias)
        
        self.classifier = self.accelerator.prepare_model(self.classifier)

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.label_smoothing = label_smoothing
        self.gamma = gamma
        self.power = power 
        self.weigh_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov


    @torch.inference_mode()
    def eval(self, 
             X: torch.Tensor, 
             Y: torch.Tensor):

        loader =  DataLoader(dataset=TensorDataset(X, Y), 
                             batch_size=self.batch_size, 
                             shuffle=False, 
                             num_workers=0,
                             drop_last=False)

        loader = self.accelerator.prepare(loader)

        corrects = torch.tensor(0, device=self.device)
        total    = torch.tensor(0, device=self.device) 

        for x, y in loader:

            out = self.classifier(x)   
            _, preds = torch.max(out, dim=1)
            corrects += torch.sum(preds == y)
            total += len(x)
        
        corrects = self.accelerator.reduce(corrects)
        total    = self.accelerator.reduce(total)

        accuracy = corrects/total

        return accuracy.cpu().item()


    
    def fit(self, 
            X: torch.Tensor, 
            Y: torch.Tensor,
            X_eval: torch.Tensor = None,
            Y_eval: torch.Tensor = None):


        loader =  DataLoader(dataset=TensorDataset(X, Y), 
                             batch_size=self.batch_size, 
                             shuffle=True, 
                             num_workers=cpu_count(),
                             drop_last=True)

        optimizer = SGD(self.classifier.parameters, 
                        lr=self.lr, 
                        momentum=self.momentum, 
                        weight_decay=self.weigh_decay, 
                        nesterov=self.nesterov)
        
        steps_one_epoch = len(X) // self.batch_size

        lr_scheduler = ExponentialScheduler(optimizer=optimizer, 
                                         max_steps=self.epochs * steps_one_epoch, 
                                         gamma=self.gamma, 
                                         power=self.power)


        loader, optimizer, lr_scheduler = self.accelerator.prepare(loader, 
                                                                optimizer, 
                                                                lr_scheduler)

        criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)


        for epoch in range(self.epochs):
            
            self.accelerator.print(f"Starting EPOCH {epoch+1}/{self.epochs}")
            for x, y in loader:
            
                logits = self.classifier(x)
                loss = criterion(logits, y)

                self.accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
         
            if X_eval is not None and Y_eval is not None:
                accuracy = self.eval(X_eval, Y_eval)
                self.accelerator.print(f"Eval accuracy: {100*accuracy:.2f}%")
        
