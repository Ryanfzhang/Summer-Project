# Chlorophyll-a Prediction
## Background
Chlorophyll-a is an essential indicator of ocean primary productivity<sup>[1]</sup>. Future Chlorophyll-a trends is important for analysis of Pearl River Estuary.
## Problem Setting
Predict the next moment's chlorophyll concentration using the past twelve time-steps.
|    Previous Observation (1-12th) | Future (13th) | 
|  ----------- |----------- |
|  <img src="https://github.com/Ryanfzhang/Summer-Project/assets/150044070/5fa358d3-4c88-4869-b490-2eafbaa2335c" width="300" height="200"/>|<img src="https://github.com/Ryanfzhang/Summer-Project/assets/150044070/4831b45e-0a03-4f8d-943d-9fb789725d81" width="300" height="200"/>|




## Dataset and Illustration
Dataset is available at https://drive.google.com/file/d/1imsmXtD-oqbAibckTY0J0A6mcgHjUF1-/view?usp=sharing.

To read data:
```python
import pickle as pkl
with open("./data/dataset.pk", "rb") as f:
  train_datas, train_labels, train_label_masks, test_datas, test_labels, test_label_masks = pkl.load(f)
```


[1] Ye H, Tang S, Yang C. Deep learning for Chlorophyll-a concentration retrieval: A case study for the Pearl River estuary[J]. Remote Sensing, 2021, 13(18): 3717.
