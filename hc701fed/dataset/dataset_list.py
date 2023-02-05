from hc701fed.dataset.EyePACS_and_APTOS import Eye_APTOS
from hc701fed.dataset.messidor import MESSIDOR

from torch.utils.data import ConcatDataset

Eye_APTOS_data_dir_options = {
    'EyePACS': '/home/xiangjian.hou/hc701-fed/preprocessed/eyepacs',
    'APTOS': '/home/xiangjian.hou/hc701-fed/preprocessed/aptos',
}


MESSIDOR_data_dir_options = {
    'messidor2': '/home/xiangjian.hou/hc701-fed/preprocessed/messidor2',
    'messidor_pairs' : '/home/xiangjian.hou/hc701-fed/preprocessed/messidor/messidor_pairs',
    'messidor_Etienne' : '/home/xiangjian.hou/hc701-fed/preprocessed/messidor/messidor_Etienne',
    'messidor_Brest-without_dilation' : '/home/xiangjian.hou/hc701-fed/preprocessed/messidor/messidor_Brest-without_dilation'
}


APTOS_train = Eye_APTOS(data_dir=Eye_APTOS_data_dir_options['APTOS'], train=True, transform=None)
EyePACS_train = Eye_APTOS(data_dir=Eye_APTOS_data_dir_options['EyePACS'], train=True, transform=None)
MESSIDOR_2_train = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor2'], train=True, transform=None)
MESSIDOR_pairs_train = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor_pairs'], train=True, transform=None)
MESSIDOR_Etienne_train = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor_Etienne'], train=True, transform=None)
MESSIDOR_Brest_train = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor_Brest-without_dilation'], train=True, transform=None)

Centerlized_train = ConcatDataset([APTOS_train, EyePACS_train, MESSIDOR_2_train, MESSIDOR_pairs_train, MESSIDOR_Etienne_train,MESSIDOR_Brest_train])
MESSIDOR_Centerlized_train = ConcatDataset([MESSIDOR_pairs_train, MESSIDOR_Etienne_train,MESSIDOR_Brest_train])

APTOS_test = Eye_APTOS(data_dir=Eye_APTOS_data_dir_options['APTOS'], train=False, transform=None)
EyePACS_test = Eye_APTOS(data_dir=Eye_APTOS_data_dir_options['EyePACS'], train=False, transform=None)
MESSIDOR_2_test = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor2'], train=False, transform=None)
MESSIDOR_pairs_test = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor_pairs'], train=False, transform=None)
MESSIDOR_Etienne_test = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor_Etienne'], train=False, transform=None)
MESSIDOR_Brest_test = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor_Brest-without_dilation'], train=False, transform=None)

Centerlized_test = ConcatDataset([APTOS_test, EyePACS_test, MESSIDOR_2_test, MESSIDOR_pairs_test, MESSIDOR_Etienne_test,MESSIDOR_Brest_test])
MESSIDOR_Centerlized_test = ConcatDataset([MESSIDOR_pairs_test, MESSIDOR_Etienne_test,MESSIDOR_Brest_test])