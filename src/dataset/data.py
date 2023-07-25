
import torch
import pandas as pd
import numpy as np

drop_columns = ['Unnamed: 0',
'unit.IRI_KEY',
'unit.SY',
'unit.GE',
'unit.VEND',
'unit.ITEM',
'price.IRI_KEY',
'price.SY',
'price.GE',
'price.VEND',
'price.ITEM',
'price.cate',
'F.IRI_KEY',
'F.SY',
'F.GE',
'F.VEND',
'F.ITEM',
'F.cate',
'D.IRI_KEY',
'D.SY',
'D.GE',
'D.VEND',
'D.ITEM',
'D.cate',
'holiday.IRI_KEY',
'holiday.SY',
'holiday.GE',
'holiday.VEND',
'holiday.ITEM',
'holiday.cate',]

def read_csv(i):
    # read each slots csv, create tensor of dimension:
    # (num_timeseries, lenght, features)
    dataset = []
    data = pd.read_csv(f"src\dataset\iri{i}.csv", usecols=lambda x: x not in drop_columns)

    cl = data.columns.to_list()
    clcat = ['unit.cate'] 
    clu = [c for c in cl if 'unit.1' in c]
    clp = [c for c in cl if 'price.1' in c]
    clh = [c for c in cl if 'holiday.1' in c]
    clf = [c for c in cl if 'F.' in c]
    cld = [c for c in cl if 'D.' in c]

    
    dataset.append(data[clu].values)
    #replace some inf values in price
    data_clp = data[clp].replace(np.inf, np.nan).interpolate()
    dataset.append(data_clp.values)
    dataset.append(data[clh].values)
    dataset.append(data[clf].values)
    dataset.append(data[cld].values)
    dataset = np.array(dataset)
    dataset = np.transpose(dataset,(1,2,0))
    return dataset , data[clcat].values

def normalize(d):
    # normalize unit and price for NNs
    norm = torch.nn.InstanceNorm1d(2)
    dd_norm = norm(d[:,:,:2])
    d[:,:,:2] = dd_norm
    return d

def concat_slots(fist_slot, last_slot):
    # concat slots to create whole train/valid/test dataset 
    dataset = []
    catg = []
    for i in range(fist_slot, last_slot):
        tens , cat = read_csv(i)
        dataset.append(tens)
        catg.append(cat)

    catg = np.concatenate(catg,axis=0)
    dataset = np.concatenate(dataset,axis=0)
    d_troch = torch.Tensor(dataset)
    d_troch_normalized = normalize(d_troch)    
    return dataset , d_troch_normalized[:20,:,:], catg #np.array , torch.array, np.array

def mode_indx(mode):
    if mode == "train":
        (start,end) = (1,8) #(60993, 55, 5)
    if mode == "valid":
        (start,end) = (8,11)#(22951, 55, 5)
    if mode == "test":     
        (start,end) = (11,16)#(36194, 55, 5)
    return (start,end)

class Dataset():
    def __init__(self, mode):
        (star,end) = mode_indx(mode)
        self.x, self.x_norm, self.catg = concat_slots(star,end)            
