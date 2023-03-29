from hc701fed.dataset.EyePACS_and_APTOS import Eye_APTOS
from hc701fed.dataset.messidor import MESSIDOR

from hc701fed.dataset.WeightedConcatDataset import WeightedConcatDataset

import os
import yaml

from hc701fed.transform.transforms import compose
from torchvision.transforms import transforms
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    interpolation = InterpolationMode.BICUBIC
except:
    import PIL
    interpolation = PIL.Image.BICUBIC

PATH_DATA = os.getcwd()

train_transforms = transforms.Compose([
    transforms.Lambda(lambda x: Image.fromarray(x)),
    transforms.RandomResizedCrop(224, scale=(0.67, 1.0), interpolation=interpolation),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3778, 0.2800, 0.2310], std=[0.2892, 0.1946, 0.2006]),
])

test_transforms = transforms.Compose([
    transforms.Lambda(lambda x: Image.fromarray(x)),
    transforms.Resize(224, interpolation=interpolation),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3778, 0.2800, 0.2310], std=[0.2892, 0.1946, 0.2006]),
])


PATH_DATA = os.getcwd()



Eye_APTOS_data_dir_options = {
    'EyePACS': os.path.join(PATH_DATA, 'preprocessed/eyepacs'),
    'APTOS': os.path.join(PATH_DATA, 'preprocessed/aptos'),
}

MESSIDOR_data_dir_options = {
    'messidor2': os.path.join(PATH_DATA, 'preprocessed/messidor2'),
    'messidor_pairs' : os.path.join(PATH_DATA, 'preprocessed/messidor/messidor_pairs'),
    'messidor_Etienne' : os.path.join(PATH_DATA, 'preprocessed/messidor/messidor_Etienne'),
    'messidor_Brest-without_dilation' : os.path.join(PATH_DATA, 'preprocessed/messidor/messidor_Brest-without_dilation')
}

APTOS_Val = Eye_APTOS(data_dir=Eye_APTOS_data_dir_options['APTOS'], mode='val', transform_=test_transforms)
EyePACS_Val = Eye_APTOS(data_dir=Eye_APTOS_data_dir_options['EyePACS'], mode='val', transform_=test_transforms)
MESSIDOR_2_Val = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor2'], mode='val', transform_=test_transforms)
MESSIDOR_pairs_Val = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor_pairs'], mode='val', transform_=test_transforms)
MESSIDOR_Etienne_Val = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor_Etienne'], mode='val', transform_=test_transforms)
MESSIDOR_Brest_Val = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor_Brest-without_dilation'], mode='val', transform_=test_transforms)

APTOS_Test = Eye_APTOS(data_dir=Eye_APTOS_data_dir_options['APTOS'], mode='test', transform_=test_transforms)
EyePACS_Test = Eye_APTOS(data_dir=Eye_APTOS_data_dir_options['EyePACS'], mode='test', transform_=test_transforms)
MESSIDOR_2_Test = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor2'], mode='test', transform_=test_transforms)
MESSIDOR_pairs_Test = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor_pairs'], mode='test', transform_=test_transforms)
MESSIDOR_Etienne_Test = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor_Etienne'], mode='test', transform_=test_transforms)
MESSIDOR_Brest_Test = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor_Brest-without_dilation'], mode='test', transform_=test_transforms)

# # Centerlized_Test = ConcatDataset([APTOS_Test, EyePACS_Test, MESSIDOR_2_Test, MESSIDOR_pairs_Test, MESSIDOR_Etienne_Test,MESSIDOR_Brest_Test])
# MESSIDOR_Centerlized_Test = ConcatDataset([MESSIDOR_pairs_Test, MESSIDOR_Etienne_Test,MESSIDOR_Brest_Test])