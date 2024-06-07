# Chlorophyll-a Prediction
## Background
Chlorophyll-a is an essential indicator of ocean primary productivity<sup>[1]</sup>. Future Chlorophyll-a trends is important for analysis of Pearl River Estuary.
## Problem Setting
Predict the next moment's chlorophyll concentration using the past twelve time-steps.
|    Previous Observation (1-12th) | Future (13th) | 
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
- The shape of **training labels** is $B\times N\times T' \times H\times W$, where $T'=1$ represents the timesteps of prediction.
- The shape of **training label masks** is one-hot tensor whose shape is the same as **training labels**. Due to the missing observations of chlorophyll data, each value in **training label masks** represents whether the data in the **training labels**  is obtained from real observations (1 represents true).

data/is_sea.npy:
```python
import numpy as np
with open("./data/dataset.pk", "rb") as f:
    train_datas, train_labels, train_label_masks, test_datas, test_labels, test_label_masks = pkl.load(f)
```


[1] Ye H, Tang S, Yang C. Deep learning for Chlorophyll-a concentration retrieval: A case study for the Pearl River estuary[J]. Remote Sensing, 2021, 13(18): 3717.
