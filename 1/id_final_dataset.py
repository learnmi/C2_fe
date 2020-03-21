import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt

# the library we will use to create the model 
from sklearn import linear_model
from sklearn.linear_model import Lasso

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures

from sklearn import metrics
import math

DATA_FILE       = "/Users/sonu/Documents/aiml/assignments/c2_m1/1/final_dataset.csv"
DATA_SEP        = ','
NUM_FEATURES    = 13

def get_mean_value(x):
    return np.mean(x)

def convert_floor_to_ratio(x):
    if isinstance(x, str):
        # check string
        if x.strip() == 'None' or x.strip() == '':
            return None
        pat_1 = '^([0-9]*)[a-zA-Z ]+([0-9]*)[a-zA-Z ]+$'
        pat_2 = '^([a-zA-Z]*)[a-zA-Z ]+([0-9]*)[a-zA-Z ]+$'
        m = re.match(pat_1, x.strip())
        if m:
            if m[1] == '':
                m_ = re.match(pat_2, x.strip())
                if m_:
                    if m_[1]=='Ground':
                        n = 0
                    else:
                        n = -1
            else:
                n = float(m[1])
            
            d = float(m[2])
            return n/d
        else:
            return 0
    else:
        return x

def convert_int(x):
    if isinstance(x, str):
        # check string
        if x.strip() == 'None' or x.strip() == '':
            return None
        int_pat = '^([0-9,]+)\s([a-zA-Z]+)'
        m = re.match(int_pat, x)
        if m:
            sqftVal = m[1].replace(',','')
            unit    = m[2]
            if unit == 'sqm':
                sqftVal = float(sqftVal)*10.7639
            return float(sqftVal)
        else:
            None
        return None
    else:
        return x

def convert_to_lakh(x):
    if isinstance(x, str):
        # check string
        if x.strip() == 'None' or x.strip() == '':
            return None
        price_pat = '^.?\s([0-9]*.?[0-9]*)?\s([a-zA-Z]+)$'
        m = re.match(price_pat, x.strip())
        if m:
            cx = float(m[1]) * 100 if m[2] == 'Cr' else float(m[1])
            return cx
        else:
            return None
    else:
        return x

def process_data(x):
    if isinstance(x, str):
        # check string
        if x.strip() == 'None' or x.strip() == '':
            return None
        return x
    elif math.isnan(x):
        return None
    else:
        return x

# read inputs
df = pd.read_csv(DATA_FILE, sep=DATA_SEP)
origdf = pd.read_csv(DATA_FILE, sep=DATA_SEP)

print(df.head())
print(df.columns)

# 3 Remove columns with missing values > 40% of size
numColumns = len(df.columns)
numRows =  df.size/numColumns
cols = df.columns

colThreshold = .4 * numRows

for column in cols:
    columnValues = df[column]
    sparse = [x for x in columnValues if process_data(x) == None]
    if len(sparse) > colThreshold:
        _ = df.pop(column)
        # print('Removed : ', column, 'total Nones : ', len(sparse))

# Result - 3
print(df.head())

print(origdf.head())
import re

# Result - 4
# Extracting 'property_price and replacing missing value mean
prop = origdf['property_price '].values.reshape(-1, 1)
print(prop)
prop_clean  = [convert_to_lakh(x[0]) for x in prop if process_data(x[0]) != None]

prop_p = []
for x in prop:
    if process_data(x[0])!= None:
        prop_p.append(convert_to_lakh(x[0]))
    else:
        prop_p.append(get_mean_value(prop_clean))
print(prop_p)

# Extracting ' square_area' and replacing missing value with mean
sa = origdf['super area    '].values.reshape(-1, 1)
sa_clean = [convert_int(x[0]) for x in sa if process_data(x[0]) != None]
print(sa_clean)
sa_p = []
for x in sa:
    if process_data(x[0]) != None:
        sa_p.append(convert_int(x[0]))
    else:
        sa_p.append(get_mean_value(sa_clean))
print(sa_p)

# Extracting ' carpet area' and replacing missing value with mean
carp = origdf['carpet area '].values.reshape(-1, 1)
carp_clean = [convert_int(x[0]) for x in carp if process_data(x[0]) != None]
print(carp_clean)
carp_p = []
for x in carp:
    if process_data(x[0]) != None:
        carp_p.append(convert_int(x[0]))
    else:
        carp_p.append(get_mean_value(carp_clean))

print(carp_p)

# Result 5, prop_p, sa_p & carp_p has values filled.
# carp_p
# sa_p
# prop_p

# Result 6&7, facing
from enum import IntEnum
class Facing(IntEnum):
    NotAvailable = 0
    North_East = 1
    East = 2
    North = 3
    South_East = 4
    South = 5
    Error = 6

def convert_facing(x):
    if x.strip() == 'North - East':
        return Facing.North_East
    elif x.strip() == 'East':
        return Facing.East
    elif x.strip() == 'North':
        return Facing.North
    elif x.strip() == 'South - East':
        return Facing.South_East
    elif x.strip() == 'South':
        return Facing.South
    elif x.strip() == '':
        return Facing.NotAvailable
    else:
        return Facing.Error

facing = origdf['facing       ']
facing_p = []
for x in facing:
    if process_data(x) != None:
        facing_p.append(int(convert_facing(x)))
    else:
        facing_p.append(int(Facing.NotAvailable))
print(facing_p)

# Result 6 & 7, overlooking
from enum import IntEnum
class Overlooking(IntEnum):
    NotAvailable = 0    
    Garden_Park_Main_Road   = 1
    Garden_Park = 2
    Garden_Park_Pool_Main_Road = 3
    Garden_Park_Pool = 4
    Main_Road = 5
    Pool = 6
    Pool_Garden_Park = 7
    Error = 8

def convert_overlooking(x):
    if x.strip() == 'Garden/Park, Main Road':
        return Overlooking.Garden_Park_Main_Road
    elif x.strip() == 'Garden/Park':
        return Overlooking.Garden_Park
    elif x.strip() == 'Garden/Park, Pool, Main Road':
        return Overlooking.Garden_Park_Pool_Main_Road
    elif x.strip() == 'Garden/Park, Pool':
        return Overlooking.Garden_Park_Pool
    elif x.strip() == 'Main Road':
        return Overlooking.Main_Road
    elif x.strip() == 'Pool':
        return Overlooking.Pool
    elif x.strip() == 'Pool, Garden/Park':
        return Overlooking.Pool_Garden_Park
    elif x.strip() == '':
        return Overlooking.NotAvailable
    else:
        return Overlooking.Error

overlooking = origdf['overlooking                    ']
overlooking_p = []
for x in overlooking:
    if process_data(x) != None:
        overlooking_p.append(int(convert_overlooking(x)))
    else:
        overlooking_p.append(int(Overlooking.NotAvailable))
print(overlooking_p)

from enum import IntEnum
class Ownership(IntEnum):
    NotAvailable = 0
    Leasehold = 1    
    freehold = 2
    Error = 3

def convert_own(x):
    if x.strip() == 'Freehold':
        return Ownership.freehold
    elif x.strip() == 'Leasehold':
        return Ownership.Leasehold
    elif x.strip() == '':
        return Ownership.NotAvailable
    else:
        return Ownership.Error

own = origdf['ownership ']
own_p = []
for x in own:
    if process_data(x) != None:
        own_p.append(int(convert_own(x)))
    else:
        own_p.append(int(Ownership.NotAvailable))
print(own_p)

from enum import IntEnum
class Tx(IntEnum):
    NotAvailable = 0
    NewProperty = 1
    Resale  = 2
    Error = 3

def convert_tx(x):
    if x.strip() == 'New Property':
        return Tx.NewProperty
    elif x.strip() == 'Resale':
        return Tx.Resale
    elif x.strip() == '':
        return Tx.NotAvailable
    else:
        return Tx.Error

tx = origdf['transaction  ']
tx_p = []
for x in tx:
    if process_data(x) != None:
        tx_p.append(int(convert_tx(x)))
    else:
        tx_p.append(int(Tx.NotAvailable))
print(tx_p)

from enum import IntEnum
class Fur(IntEnum):
    NotAvailable = 0
    SemiFurnished = 1
    Unfurnished  = 2
    Furnished = 3
    Error = 4

def convert_fur(x):
    if x.strip() == 'Semi-Furnished':
        return Fur.SemiFurnished
    elif x.strip() == 'Unfurnished':
        return Fur.Unfurnished
    elif x.strip() == 'Furnished':
        return Fur.Furnished
    elif x.strip() == '':
        return Fur.NotAvailable
    else:
        return Tx.Error

f = origdf['furnishing']
f_p = []
for x in f:
    if process_data(x) != None:
        f_p.append(int(convert_fur(x)))
    else:
        f_p.append(int(Fur.NotAvailable))
print(f_p)
####, , 

#Result 8
floorR = origdf['floor                   ']
floorR_p = []
for x in floorR:
    if process_data(x) != None:
        floorR_p.append(convert_floor_to_ratio(x))
    else:
        floorR_p.append(0)
print(floorR_p)

#Result 9
bal = origdf['balcony ']
bal_p = []
for x in bal:
    if process_data(x.strip()) != None:
        bal_p.append(float(x.strip()))
    else:
        bal_p.append(0)
print(bal_p)

# Final result, merging all column vectors
finaldf = pd.DataFrame({
                'balcony': bal_p,
                'floor': floorR_p,
                'overlooking': overlooking_p,
                'facing': facing_p,
                'carpet area': carp_p,
                'super area': sa_p,
                'property price': prop_p,
                'ownership': own_p,
                'transaction': tx_p,
                'furnished': f_p
                })
print(finaldf.head())
