# gcn_brain_decoding
Project for the deep learning class 

# Running models on a Slurm cluster

Depending on which parameter you wish to optimize (loss, number of graph filters, etc.) use the following command:
`sbatch --array=1-$(sed -n '$=' model_args_[parameter].txt) -o ./logs/ModelTraining-%j.out --job-name=ModelTraining -p short --constraint="skl-compat" --cpus-per-task=2 --requeue job_gcn_train.sh model_args_[parameter].txt`

Each parameter has a separate .txt file which contains arguments which are identical except for the one we are trying to optimize. E.g., in `model_args_loss.txt`, loss function changes each time we train a model. 

