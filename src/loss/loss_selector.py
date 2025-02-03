from torch.nn import CrossEntropyLoss

class LossSelector:
    def __init__(self, loss_name, weight=None):
        self.loss_name = loss_name
        self.weight = weight
        self.loss = self._select_loss()

    def _select_loss(self):
        if self.loss_name == 'cross_entropy':
            if self.weight is None:
                return CrossEntropyLoss()
            else:
                return CrossEntropyLoss(weight=self.weight)
        else:
            raise ValueError('Invalid loss name')
        
    def get_loss(self):
        return self.loss