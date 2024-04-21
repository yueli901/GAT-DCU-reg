# Dynamic Central Update Mechanism with Regularization in Graph Attention Networks for Traffic Forecasting Explainability


This is the Tensorflow implementation of ST-MetaNet-DCU with regularization in the following paper:

- Yue Li, ShujuanChen. Dynamic Central Update Mechanism with Regularization in Graph Attention Networks for Traffic Forecasting Explainability.

---

## Requirements for Reproducibility

### System Requirements:

- System: Windows 11 OS
- Language: Python 3.9.18
- Devices: a single RTX 4090 GPU


### Library Requirements:

- numpy 1.25.2
- pandas 2.0.3
- [TensorFlow 2.10.1](https://www.tensorflow.org/install/pip?_gl=1*1tk6s5m*_up*MQ..*_ga*MjI3MzQyMDc1LjE3MTM2OTIwNzI.*_ga_W0YLR4190T*MTcxMzY5MjA3MS4xLjAuMTcxMzY5MjA3MS4wLjAuMA..#windows-native)
- cudatoolkit 11.2.2
- cudnn 8.1.0.77
- [DGL 1.1.2+cu118](https://www.dgl.ai/pages/start.html)
- tables 3.9.1
- holidays 0.45
- pyymal 6.0
- h5py 3.10.0

Mannual installation is recommended for installing the dependencies. After installation, change the backend deep learning framework for dgl to tensorflow.

---

### Data Description
Highway trunk road traffic data are downloaded from [Highway England](http://tris.highwaysengland.co.uk/). Edge features and node features are obtained using Google Route API and Google Map.

Unzip the data files in `data/data.zip`. Move the two files below under the same directory.
- `sensor498_2019-01-01_2019-12-31.h5` is the original data downloaded.
- `sensor498_2019-01-01_2019-12-31_imputed.h5` is the imputed data. Imputation is only performed for speed measurements when both volume and speed are 0, using the closest previous non-NA speed measurement. The script for this imputation task is in `data/imputation.ipynb`. The shape of data is (498, 35040, 2), which indicates (number of sensors, number of timestamps, number of traffic features). Speed and volume are the two traffic features.

Data for other years (2015-2023) can be downloaded from this [link](https://pan.baidu.com/s/1qey-HshcizFInAYzBhQj7g?pwd=a8va). Script of downloading using API is included.

- `tris_edge_features.xlsx`: Edge features (original graph) of the highway trunk road traffic network. Two road segments do not have any valid sensor and the traffic readings are replaced with those recorded by the sensor on the other direction of the road segment, indicated with sensors id starting with '999'. The shape of edge features is (498, 10), which indicates (number of edges, number of edge features). The 10 edge features include the longitude and latitude of the sensor (2), distance of the road segment and driving duration (2), longitude and latitude of the origin and destination of the road segment (4), the number of neighbours of the origin and destination of the road segment (2).
- `tris_node_features.csv`: Node features (original graph) of the highway trunk road traffic network. The shape of node features is (181, 2), which indicates (number of nodes, number of node features). The two features are the longitude and latitude of the nodes.


### Code Description

- `data` folder stores the data, including traffic data, edge features and node features.
- `eval` folder stores the logs of model training and evaluation metrics, which are outputs of running the `train.py` script.
- `interpret` folder stores the python script used to make inference using trained models. The script first establishes the model then loads existing model parameters. In this `rho-matrix.ipynb` file, predictions of four timestamps of interest are made using regularized ST-MetaNet-DCU models with different $\lambda$. During this process the $\rho$ matrix are generated and collected for further analysis.
- `param` folder stores all trained model parameters.
- `ST-MetaNet`, `ST-MetaNet-DCU`, and `ST-MetaNet-DCU-reg` are folders that store the source code of each model. The main difference between the first two is in the `graph.py` where `MetaGAT` is modified. The main difference between the last two is the `graph.py`, `seq2seq.py`, and `train.py` where the intermediate output $\rho$ matrix are sent together with the prediction, and then participate in the loss function definition. The regularization parameter $\lambda` is also newly defined in the `model_setting/st-metanet.yaml`, which is then passed to `train.py`.
- `train.py` is the script for a single run of model training, which can be executed using a shell command like `python train.py --file model_setting/st-metanet.yaml --epochs 2` after `cd` to the corresponding model folder and the Python environment activated. `run.ipynb` is a script that can run multiple times of `train.py`, so that each single model with different hyperparameters can be trained consecutively.

---

## Citation

If you find this repository, e.g., the code and the datasets, useful in your research, please cite our paper.

For the original Mxnet implementation of the baseline ST-MetaNet model, please refer to the ![ST-MetaNet repository](https://github.com/panzheyi/ST-MetaNet) and the following paper:

- Zheyi Pan, Yuxuan Liang, Weifeng Wang, Yong Yu, Yu Zheng, and Junbo Zhang. [Urban Traffic Prediction from Spatio-Temporal Data Using Deep Meta Learning](https://www.researchgate.net/publication/333186315_Urban_Traffic_Prediction_from_Spatio-Temporal_Data_Using_Deep_Meta_Learning). 2019. In The 25th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD'19), August 4–8, 2019, Anchorage, AK, USA.


---

## License

MIT License (refer to the LICENSE file for details).