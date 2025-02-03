from src.model.model_type import DenseNet121

class ModelSelector:
    def __init__(self, model_name, weights="default", **kwargs):
        self.model_name = model_name
        self.weights = weights
        self.model = self._select_model(**kwargs)

    def _select_model(self, **kwargs):
        model_name = self.model_name.lower()
        if model_name == 'densenet121':
            return DenseNet121(weights=self.weights, **kwargs)
        else:
            raise ValueError(f"Model {model_name} not found")
        
    def get_model(self):
        return self.model.get_model()
    
    def get_model_features(self):
        return self.model.features
    
    def get_model_classifier(self):
        return self.model.classifier