# ClusterGNN Adapter



We introduce **ClusterGNN Adapter,** **a** **plug-and-play relational learning module for multivariate time series forecasting (MTF) models.** The proposed adapter dynamically captures cluster-level inter-variable dependencies through Gumbel-Softmax-based graph construction and propagation, while preserving the original forecasting architecture. The propagated relational representations are adaptively fused with the original input, enabling existing MTF backbones **to effectively incorporate global dependency information without architectural modification.**



## Overall Architecture

### [1] ClusterGNN Adapter

 The adapter can be seamlessly integrated into various forecasting backbones, such as iTransformer, PatchTST, TimeMixer, and SimpleTM, without modifying their original architectures.



![image-20260512211931972](/Figures/ClusterGNN_Adapter_Architecture.png)



### [2] ClusterGNN Module

ClusterGNN is a cluster-level graph propagation framework for modeling dynamic inter-variable dependencies in multivariate time series.

The module consists of three components:

 **(1) cluster-level Gumbel-Softmax assignment**
      : dynamically groups variables into latent clusters

 **(2) edge-level dynamic graph construction**
      : learns probabilistic inter-cluster connections

 **(3) cluster graph propagation with cluster-to-node projection**
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

Table 1 reports the forecasting performance of <u>six baseline models</u> and their <u>ClusterGNN-enhanced variants</u> across multiple benchmark datasets. **Better results between each baseline pair are highlighted in bold**, while the best overall results for each dataset are marked in red.

![image-20260512212720547](/Figures/main_results.png)

Especially, **PatchTST** shows the largest performance improvement, achieving an average **MSE reduction of 17.7%.** Figure 3 further illustrates qualitative forecasting results on the Exchange dataset, where the ClusterGNN-enhanced PatchTST produces more stable and accurate long-term predictions. 



![image-20260512211300585](/Figures/PatchTST_result.png)





## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.

- SimpleTM (https://github.com/vsingh-group/SimpleTM)
- TimeXer (https://github.com/thuml/TimeXer)
- Time-Series-Library (https://github.com/thuml/Time-Series-Library/tree/main/models ) 
- iTransformer (https://github.com/thuml/iTransformer)
- GTA (https://github.com/zackchen-lb/GTA)

---

