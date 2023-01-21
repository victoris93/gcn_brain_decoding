#!/bin/bash
args_file=$1

#SBATCH --job-name=ModelTraining
#SBATCH -o ./logs/ModelTraining-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=2
#SBATCH --requeue
#SBATCH --array=1-$(cat $args_file | wc -l)

module load Python/3.9.6-GCCcore-11.2.0
source /gpfs3/users/margulies/cpy397/env/ClinicalGrads/bin/activate

arg=$(sed -n "$((SLURM_ARRAY_TASK_ID -1))p" $args_file)

echo Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
echo the job id is $SLURM_ARRAY_JOB_ID
echo arguments: $arg
echo SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}
cat $args_file

python3 -u gcn_optimization_cluster.py $arg

