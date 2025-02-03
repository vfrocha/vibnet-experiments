from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from adabelief_pytorch import AdaBelief

class OptimizationSelector:
    def __init__(self, optimizer_name, scheduler_name, learning_rate, **kwargs):
        self.optimizer_name = optimizer_name.lower()
        self.scheduler_name = scheduler_name.lower()
        self.learning_rate = learning_rate
        self.kwargs = kwargs

        self.optimizer = None
        self.scheduler = None

    def get_optimizer(self, model_parameters):
        if self.optimizer_name == 'adam':
            self.optimizer = Adam(model_parameters, lr=self.learning_rate)
        elif self.optimizer_name == 'sgd':
            self.optimizer = SGD(model_parameters, lr=self.learning_rate)
        elif self.optimizer_name == 'adabelief':
            self.optimizer = AdaBelief(model_parameters, lr=1e-3, eps=1e-8, weight_decay=1e-2, betas=(0.9,0.999), weight_decouple = True, rectify = False)
        else:
            raise ValueError('Invalid optimization name')
        
        return self.optimizer

    def get_scheduler(self):
        if not self.optimizer:
            raise ValueError('Optimizer is not initialized')
        
        if self.scheduler_name == 'step_lr':
            step_size = self.kwargs.get('step_size', 30)
            gamma = self.kwargs.get('gamma', 0.1)
            self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif self.scheduler_name == 'reduce_lr_on_plateau':
            mode = self.kwargs.get('mode', 'min')
            factor = self.kwargs.get('factor', 0.1)
            patience = self.kwargs.get('patience', 10)
            min_lr = self.kwargs.get('min_lr', 10e-6)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr)
        else:
            raise ValueError('Invalid scheduler name')

        return self.scheduler