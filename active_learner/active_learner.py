from __future__ import print_function
from __future__ import division
import os

os.environ["CUDA_LAUNCH_BLOCKING"]="1"

import os
import sys
import time
import copy
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from gui import Application, MainWindow, Window

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def train_model(model, 
                dataloaders, 
                criterion, 
                optimizer,
                device,
                phases=['train', 'val'], 
                num_epochs=1, 
                is_inception=False):
    start_time = time.time()

    history = defaultdict(list)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    targets = torch.tensor([d[1] for d in dataloaders['train']], device=device)

    class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets)], 
        dtype=torch.float32, 
        device=device
        )
    weight = 1. / class_sample_count
    weight = weight / weight.sum()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss = loss * weight[labels.long()].to(loss.device)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.cpu().item()
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects.double() / len(dataloaders[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            history[phase].append(epoch_acc)

    #time_elapsed = time.time() - start_time
    #print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model, history

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()


    model_ft.name = model_name
    return model_ft, input_size

def initialize_preprocessor(input_size, device, phase='train'):        
    # Data augmentation and normalization for training
    # Just normalization for validation
    if phase == 'train':
        preprocessor = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Lambda(lambda image: image[None].to(device))
        ])
    elif phase in ['eval', 'test']:
        preprocessor = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Lambda(lambda image: image[None].to(device))
        ])

    return preprocessor
class ActiveLearner(torch.nn.Module):
    def __init__(self, model, input_size, num_classes, device=None, num_epochs=1, optimizer=None, learning_criterion=None):
        super(ActiveLearner, self).__init__()

        # send the model to GPU
        if device is None:
            # Detect if we have a GPU available
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        model = model.to(device)
        
        self.model = model
        
        self.train_preprocessor = initialize_preprocessor(
            input_size, device=self.device, phase="train")
        self.eval_preprocessor = initialize_preprocessor(
            input_size, device=self.device, phase="test")

        # setup the optimizer
        if optimizer is None:
            optimizer = setup_optimizer(self.model, feature_extract=True)
        self.optimizer = optimizer

        # setup the loss fxn 
        if learning_criterion is None:
            learning_criterion = nn.CrossEntropyLoss()
        self.learning_criterion = learning_criterion

        self.num_epochs = num_epochs
        self.class_names = [f"Class {c}" for c in range(num_classes)]
        #for c in range(1, num_classes-1):
        #    self.class_names += [f"Class {c}"]
        #    self.grow_class_size()

        self._previous_training_images = deque()
        self._max_saved_images = 64

    def teach(self, X, y):
        if isinstance(y, int):
            y = torch.tensor([y], dtype=torch.int64)

        if not isinstance(X, torch.Tensor):
            X_train = self.train_preprocessor(X)
        else:
            X_train = X

        self._previous_training_images.append((X_train.to(self.device), 
                                               y.long().to(self.device)))
        if len(self._previous_training_images) > self._max_saved_images:
            x = self._previous_training_images.popleft()
            print("popping.....")
            del x
        new_data = {"train": self._previous_training_images}
        if y.max() >= self.num_classes:
            self.grow_class_size()
            print(f"Old class names: {self.class_names[:-1]}")
            if len(self.class_names) < y.item():
                self.class_names += [f"Class {y.item()}"]

            print(f"Adding new class with name: {self.class_names[-1].lower()}")
            print(f"class names: {self.class_names}")

        current_class_name = self.class_names[y.cpu().detach().max().int()]
        print(f"Teaching class '{current_class_name}' to learning")
        print(f"Training on {len(self._previous_training_images)} points")

        try:
            # Train and evaluate
            self.model, hist = train_model(
                self.model, 
                new_data, 
                self.learning_criterion, 
                self.optimizer,
                device=self.device,
                phases=new_data.keys(),
                num_epochs=self.num_epochs, 
                is_inception=(self.model.name=="inception")
            )
        except RuntimeError as e:
            print(e)
            torch.cuda.empty_cache()
        return self(X)

    def __call__(self, X, y=None):
        if y is not None:
            return self.teach(X, y.long())
        else:
            X = self.eval_preprocessor(X)
            y_pred = self.model(X)
            return torch.nn.functional.softmax(y_pred, dim=-1)


    def grow_class_size(self, class_name=None):
        print(f"Growing class size from {self.num_classes} to {self.num_classes + 1}")
        if class_name is None:
            class_name = f"Class {self.num_classes}"
        self.class_names += [class_name]

        self.model.fc = self.add_units(
            self.model.fc, 
            n_new=1, 
            num_output_channels=self.num_classes+1,
            device=self.device
        )
        self.optimizer = setup_optimizer(self.model, feature_extract=True)


    def add_units(self, fc_layer, n_new, num_output_channels, device):
        # take a copy of the current weights stored in self.fcs
        weight_data = fc_layer.weight.data

        # make the new weights in and out of hidden layer you are adding neurons to
        hl_input = torch.zeros([n_new, weight_data.shape[1]]).to(device)
        nn.init.xavier_uniform_(hl_input, gain=nn.init.calculate_gain('relu'))

        # concatenate the old weights with the new weights
        new_weight = torch.cat([weight_data, hl_input], dim=0)

        # reset weight and grad variables to new size
        new_layer = nn.Linear(weight_data.shape[1], num_output_channels).to(device)

        # set the weight data to new values
        new_layer.weight.data = torch.tensor(new_weight, requires_grad=True, device=device)

        #self.optimizer.param_groups.append({'final_layer': new_layer.parameters() })

        return new_layer

    @property
    def num_classes(self):
        return self.model.fc.out_features


def setup_optimizer(model, feature_extract):
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = []
    print("Params to learn:")
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            if feature_extract:
                params_to_update.append(param)
            print("\t",name)

    # Observe that all parameters are being optimized
    return optim.SGD(params_to_update, lr=0.006, momentum=0.2)


def build_active_learner(model_name="resnet",
                         num_classes=1, 
                         use_pretrained=True, 
                         feature_extract=True, **active_learner_kwargs):
    model, input_size = initialize_model(
        model_name=model_name, 
        num_classes=num_classes, 
        feature_extract=feature_extract, 
        use_pretrained=use_pretrained
    )

    learner = ActiveLearner(model=model, input_size=input_size, 
                            num_classes=num_classes, **active_learner_kwargs)
    return learner

