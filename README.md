# Chlorophyll-a Prediction
## Background
Chlorophyll-a is an essential indicator of ocean primary productivity<sup>[1]</sup>. Future Chlorophyll-a  is important for analysis of Pearl River Estuary.
## Problem Setting
Predict the next moment's chlorophyll concentration using the past twelve time-steps in Pearl River Estuary.
|    Past Observation (1-12th) | Future Chlorophyll-a we need to predict (13th) | 
|  ----------- |----------- |
|  <img src="https://github.com/Ryanfzhang/Summer-Project/assets/150044070/5fa358d3-4c88-4869-b490-2eafbaa2335c" width="400" height="250"/>|<img src="https://github.com/Ryanfzhang/Summer-Project/assets/150044070/4831b45e-0a03-4f8d-943d-9fb789725d81" width="400" height="250"/>|




## Dataset and Illustration
Dataset is available at https://drive.google.com/file/d/1imsmXtD-oqbAibckTY0J0A6mcgHjUF1-/view?usp=sharing.

To read data:
```python
import pickle as pkl
with open("./data/dataset.pk", "rb") as f:
    train_datas, train_labels, train_label_masks, test_datas, test_labels, test_label_masks = pkl.load(f)
```

Due to the frequent occurrence of missing observations in chlorophyll data, we conducted data imputation in advance to address the missing values. Hence, we obtained twenty imputed datasets. Specifically,

- The shape of **traing data** is $B\times N\times T \times H\times W$, where $B$ represents the num of samples, $N=20$ represents different dataset, $T=12$ represents the timesteps of previous observation, $H=60$ and $W=96$ represents the height and width of this area respectively.
- The shape of **training labels** is $B\times N'\times T' \times H\times W$, where $T'=1$ represents the timesteps of target. For different datasets, our target is the true observation without any imputation, so $N'=1$.
- The shape of **training label masks** is one-hot tensor whose shape is the same as **training labels**. **Training labels** are the true observations, so there are many missing values. Each value in **training label masks** represents whether the data in the **training labels**  is obtained from real observations (1 represents real observation).

data/is_sea.npy:
```python
import numpy as np
is_sea = np.load("./data/is_sea.npy")
```
Not the entire area of 60*96 is a sea area. This tensor indicates which positions are sea areas (1 represents sea).

## Results Visualization

You can obtain the longitude and latitude of each position from data/modis_chla_8d_4km_pre.mat
```python
from mpl_toolkits import basemap
import h5py
from numpy import meshgrid

tmp =prediction[0,0,0]
tmp[~is_sea.astype(bool)]=np.nan
raw_data = h5py.File("./data/modis_chla_8d_4km_pre.mat", 'r')
lon = np.array(raw_data['longitude']).squeeze()
lati = np.array(raw_data['latitude']).squeeze()

[x,y] = meshgrid(lon, lati)

lon1, lon2, lati1, lati2 = 111, 117, 20, 24
map = basemap.Basemap(llcrnrlon=lon1, llcrnrlat=lati1,urcrnrlon=lon2, urcrnrlat=lati2, projection='cyl', resolution='h')
map.fillcontinents(color='grey')
map.scatter(x, y, c=tmp, cmap="seismic")
map.colorbar(ticks=np.linspace(-1.5, 1.5, 20))
```


[1] Ye H, Tang S, Yang C. Deep learning for Chlorophyll-a concentration retrieval: A case study for the Pearl River estuary[J]. Remote Sensing, 2021, 13(18): 3717.
