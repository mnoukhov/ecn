#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=1
#SBATCH --array=0-4
#SBATCH --output=/home/noukhovm/scratch/slurm-logs/ecn.%A.%a.out
#SBATCH --error=/home/noukhovm/scratch/slurm-logs/ecn.%A.%a.err
#SBATCH --job-name=ecn
#SBATCH --mem=8GB
#SBATCH --time=7:59:00

module load python/3.7
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

export PYTHONUNBUFFERED=1

mkdir -p $SLURM_TMPDIR/wandb

python main.py \
    --name 'self-both-seed$SLURM_ARRAY_TASK_ID' \
    --savedir "$SLURM_TMPDIR/wandb" \
    --noprosocial \
    --proposal \
    --linguistic \
    --enable_cuda \
    --term-entropy-reg 0.05 \
    --utterance-entropy-reg 0.001 \
    --proposal-entropy-reg 0.05 \
    --episodes 50000 \
    --seed $SLURM_ARRAY_TASK_ID

rsync -a $SLURM_TMPDIR/wandb/ $SCRATCH/ecn/wandb/
rm -rf $SLURM_TMPDIR/env
rm -rf $SLURM_TMPDIR/wandb
