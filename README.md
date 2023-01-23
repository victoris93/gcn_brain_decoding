# Decoding Task from fMRI Activity with GCN
### Authors & Contributors
Model first implemented in [Zhang et al. (2021)](https://linkinghub.elsevier.com/retrieve/pii/S1053811921001245)
Contributors:

[Sébastien Labbé](https://github.com/SebastienLabbe)    
[Victoria Shevchenko](https://github.com/victoris93)    
## I. Different Graph Construction Method

Model performance depends on the connectivity matrix used for graph construction. Prior to the main optimization pipeline, we tried:
1. a different subject
2. z-transforming the connectivity matrix as done [here](https://github.com/zhangyu2ustc/gcn_tutorial_test).

For the original graph construction method, refer to `gcn_decoding.ipynb`. Notebooks `gcn_decoding_2.ipynb` and `gcn_decoding_s2_z.ipynb` illustrate steps 1. and 2. To visualize and compare performance metrics for all three graph construction methods, see `compare_graph_construction.ipynb`.

## II. Parallel Model Training on Cluster

We optimized each argument separately:
- loss function
- optimizer
- N of fully connected neurons
- N of graph filters at each Cheb convolution layer
- N of output channels
- Batch size
- N of epochs
- Learning rate

We used a computer cluster to train ~ 10 models in parallel for each parameter.

The initial model is described [here](https://main-educational.github.io/brain_encoding_decoding/gcn_decoding.html)

## III. Running models on a Slurm cluster

Depending on the parameter you wish to optimize (loss, number of graph filters, etc.) use the following command:
`sbatch --array=1-$(sed -n '$=' model_args_[parameter].txt) -o ./logs/ModelTraining-%j.out --job-name=ModelTraining -p short --constraint="skl-compat" --cpus-per-task=2 --requeue job_gcn_train.sh model_args_[parameter].txt`

Each parameter has a separate .txt file where only the concerned parameter changes. E.g., in `model_args_loss.txt`, loss function changes each time we train a model, but the rest of parameters remain the same. 

## References
Zhang, Y., Tetrel, L., Thirion, B., & Bellec, P. (2021). Functional annotation of human cognitive states using deep graph convolution. NeuroImage, 231, 117847.