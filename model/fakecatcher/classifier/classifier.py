from basecnn import baseCNN
from resnet import ResNet34
from effinet import EfficientNetB3


class Model:
    @staticmethod
    def get_model(model_name):
        if model_name == 'baseCNN':
            return baseCNN()
        elif model_name == 'ResNet34':
            return ResNet34()
        elif model_name == 'EfficientNetB3':
            return EfficientNetB3()
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        