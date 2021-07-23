import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet_Mod(nn.Module):
    def __init__(self, model):
        super(ResNet_Mod, self).__init__()
        
        # loading the model
        self.res = model   
        # accessing the last convolutional layer     
        self.first_part = nn.Sequential(*list(self.res.children())[:5])  #nn.Sequential(*list(loaded_model.children())[0:5])   #8x8 output
        # accessing the last classifier layer
        self.second_part = nn.Sequential(*list(self.res.children())[5:-1]) # 8x8 output
        self.max_pool = nn.MaxPool2d(kernel_size=4, stride=1, padding=0, dilation=1)
        self.classifier =  nn.Sequential(*list(self.res.children())[-1:])

        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.first_part(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        x = self.second_part(x)
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.first_part(x)
