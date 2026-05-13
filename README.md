# ClusterGNN Adapter

We introduce **ClusterGNN Adapter, a plug-and-play relational learning module for multivariate time series forecasting (MTF) models.** The proposed adapter dynamically captures cluster-level inter-variable dependencies through Gumbel-Softmax-based structure learning and graph propagation, while preserving the original forecasting architecture. The propagated relational representations are adaptively fused with the original input, enabling existing MTF backbones to incorporate global dependency information **without architectural modification.**


## Overall Architecture

### [1] ClusterGNN Adapter

 The adapter can be seamlessly integrated into various forecasting backbones, such as PatchTST, iTransformer, TimeMixer, and SimpleTM, **without modifying their original architectures.**



![image-20260512211931972](/Figures/ClusterGNN_Adapter_Architecture.png)



### [2] ClusterGNN Module

ClusterGNN is a cluster-level graph propagation framework for modeling dynamic inter-variable dependencies in multivariate time series.

The module consists of three components:

 **(1) cluster-level Gumbel-Softmax assignment**
      : dynamically groups variables into latent clusters

 **(2) edge-level dynamic graph construction**
      : learns probabilistic inter-cluster connections

 **(3) cluster graph propagation and cluster-to-node projectionn**
      : propagates relational information and restores it to the original variable space

![image-20260512211956790](/Figures/ClusterGNN_Architecture.png)

## Usage

1. The datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view)

2. Train and evaluate the model. We provide all the above tasks under the folder ./scripts/. You can reproduce the results as the following examples:

**All Model & All Datasets**

```
bash ./exp_all.sh
```

**One Model Scipts** (Example Model: SimpleTM)

```
bash ./scripts/multivariate_forecasting/Exchange_scripts/SimpleTM.sh
bash ./scripts/multivariate_forecasting/Weather_scripts/SimpleTM.sh
bash ./scripts/multivariate_forecasting/ECL_scripts/SimpleTM.sh
bash ./scripts/multivariate_forecasting/Solar_scripts/SimpleTM.sh
```



## Main Results

Table 1 reports the forecasting performance of <u>six baseline models</u> and their <u>ClusterGNN-enhanced variants</u> across multiple benchmark datasets. **Better results within each baseline pair are highlighted in bold**, while the best overall results for each dataset are marked in red.

![image-20260512212720547](/Figures/main_results.png)

In particular, **PatchTST** shows the largest performance improvement, achieving an Average **17.7% MSE reduction**. Figure 3 further illustrates qualitative forecasting results on the Exchange dataset, where the PatchTST enhanced with ClusterGNN produces more stable and accurate long-term predictions. 



![image-20260512211300585](/Figures/PatchTST_result.png)





## Acknowledgement

We gratefully acknowledge the following open-source repositories for providing valuable implementations and experimental foundations.
- SimpleTM (https://github.com/vsingh-group/SimpleTM)
- TimeXer (https://github.com/thuml/TimeXer)
- Time-Series-Library (https://github.com/thuml/Time-Series-Library/tree/main/models ) 
- iTransformer (https://github.com/thuml/iTransformer)
- GTA (https://github.com/zackchen-lb/GTA)

---

