import timm

print(timm.list_models('*resnet*'))
print(timm.list_models('*efficientnet*'))
print(timm.list_models('*mobilenet*'))
print(timm.list_models('*densenet*'))
print(timm.list_models('*vit*'))
print(timm.list_models('*vgg*'))