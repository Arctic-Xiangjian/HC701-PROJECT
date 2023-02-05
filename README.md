# HC701-PROJECT

## Experiment notification
please fix the random seed by this, and if you have time please test from 42-46.
```
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)
torch.manual_seed(42)
```

## Train

`python train.py --backbone densenet121 --batch_size 1 --epochs 500 --dataset messidor --seed 42`

## Load data
The data you can find in my /l/users/xiangjian.hou/

And demo in `HC701-PROJECT/notebook/Load_Data_and_train_Demo.ipynb`

## ideal workspace dir should be
[Your workfolder]
|  
|-- HC701-PROJECT
|  
|-- preprocessed
|   |
|   |-- aptos
|   |   |
|   |   |-- train
|   |   |
|   |   |-- test
|   |
|   |-- eyepacs
|   |   |
|   |   |-- train
|   |   |
|   |   |-- test
|   |
|   |-- messidor
|   |   |
|   |   |-- ...
|   |
|   |-- messdor2
|   |   |
|   |   |-- ...

## Test data

The messidor and messidor2 do not give offical test set. So We can split 0.5/0.25/0.25 as train/val/test split.

EyePACS and APTOS we do not have the test set labels. We need upload to the website the get the result.
