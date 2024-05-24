import inspect

model = torchvision.models.resnet18() # An instance of your model.
print(inspect.getmro(model.__class__))