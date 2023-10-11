#!/bin/bash
#SBATCH -JWhole_image_no_aug            # Job name
#SBATCH -Agts-jshi33-isye                             # Charge account
#SBATCH --partition=gpXY        # put the job into the gpu partition
#SBATCH              # request exclusive allocation of resources
#SBATCH --mem=50G                # RAM per node
#SBATCH --threads-per-core=1    # do not use hyperthreads (i.e. CPUs = physical cores below)
#SBATCH --cpus-per-task=2       # number of CPUs per process

## nodes allocation
#SBATCH --nodes=2               # number of nodes
#SBATCH --gres=gpu:RTX_6000:1   # number of GPUs per node (gres=gpu:N)

#SBATCH -t600                                    # Duration of the job (Ex: 15 mins)
#SBATCH -qinferno                                # QOS name
#SBATCH -oReport-%j.out                          # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL               # Mail preferences
#SBATCH --mail-user=smou7@gatech.edu            # e-mail address for notifications

cd $HOME/scratch/Learn_augmentation            # Change to working directory created in $HOME
module load anaconda3/2022.05                   # Load module dependencies
source activate SynData

#####################30 epochs #######################
#### whole image aug with per catgy loss recorded

srun --exclusive -n 1 -N 1 --mem-per-gpu=24G python Cut_paste_v1.py --root ./Cut_paste_v1/ --data_root 'carpet' --rnd_seed 0 --num_epochs 150  --dice_weight 0.5  --lr 0.0005 --batch_size 2 --lr_schedule_step_size 20 --apply_augmentations --lam_weights 1.0 0.0 0.0  &

srun --exclusive -n 1 -N 1 --mem-per-gpu=24G python Cut_paste_v1.py --root ./Cut_paste_v1/ --data_root 'carpet' --rnd_seed 0 --num_epochs 150  --dice_weight 0.5  --lr 0.0005 --batch_size 2 --lr_schedule_step_size 20 --apply_augmentations --lam_weights 0.0 1.0 0.0  &

srun --exclusive -n 1 -N 1 --mem-per-gpu=24G python Cut_paste_v1.py --root ./Cut_paste_v1/ --data_root 'carpet' --rnd_seed 0 --num_epochs 150  --dice_weight 0.5  --lr 0.0005 --batch_size 2 --lr_schedule_step_size 20 --apply_augmentations --lam_weights 0.0 1.0 1.0  &

srun --exclusive -n 1 -N 1 --mem-per-gpu=24G python Cut_paste_v1.py --root ./Cut_paste_v1/ --data_root 'carpet' --rnd_seed 0 --num_epochs 150  --dice_weight 0.5  --lr 0.0005 --batch_size 2 --lr_schedule_step_size 20 --apply_augmentations --lam_weights 1.0 1.0 1.0  &

srun --exclusive -n 1 -N 1 --mem-per-gpu=24G python Cut_paste_v1.py --root ./Cut_paste_v1/ --data_root 'carpet' --rnd_seed 1 --num_epochs 150  --dice_weight 0.5  --lr 0.0005 --batch_size 2 --lr_schedule_step_size 20 --apply_augmentations --lam_weights 1.0 0.0 0.0  &

srun --exclusive -n 1 -N 1 --mem-per-gpu=24G python Cut_paste_v1.py --root ./Cut_paste_v1/ --data_root 'carpet' --rnd_seed 1 --num_epochs 150  --dice_weight 0.5  --lr 0.0005 --batch_size 2 --lr_schedule_step_size 20 --apply_augmentations --lam_weights 0.0 1.0 0.0  &

srun --exclusive -n 1 -N 1 --mem-per-gpu=24G python Cut_paste_v1.py --root ./Cut_paste_v1/ --data_root 'carpet' --rnd_seed 1 --num_epochs 150  --dice_weight 0.5  --lr 0.0005 --batch_size 2 --lr_schedule_step_size 20 --apply_augmentations --lam_weights 0.0 1.0 1.0  &

srun --exclusive -n 1 -N 1 --mem-per-gpu=24G python Cut_paste_v1.py --root ./Cut_paste_v1/ --data_root 'carpet' --rnd_seed 1 --num_epochs 150  --dice_weight 0.5  --lr 0.0005 --batch_size 2 --lr_schedule_step_size 20 --apply_augmentations --lam_weights 1.0 1.0 1.0  &

srun --exclusive -n 1 -N 1 --mem-per-gpu=24G python Cut_paste_v1.py --root ./Cut_paste_v1/ --data_root 'carpet' --rnd_seed 2 --num_epochs 150  --dice_weight 0.5  --lr 0.0005 --batch_size 2 --lr_schedule_step_size 20 --apply_augmentations --lam_weights 1.0 0.0 0.0  &

srun --exclusive -n 1 -N 1 --mem-per-gpu=24G python Cut_paste_v1.py --root ./Cut_paste_v1/ --data_root 'carpet' --rnd_seed 2 --num_epochs 150  --dice_weight 0.5  --lr 0.0005 --batch_size 2 --lr_schedule_step_size 20 --apply_augmentations --lam_weights 0.0 1.0 0.0  &

srun --exclusive -n 1 -N 1 --mem-per-gpu=24G python Cut_paste_v1.py --root ./Cut_paste_v1/ --data_root 'carpet' --rnd_seed 2 --num_epochs 150  --dice_weight 0.5  --lr 0.0005 --batch_size 2 --lr_schedule_step_size 20 --apply_augmentations --lam_weights 0.0 1.0 1.0  &

srun --exclusive -n 1 -N 1 --mem-per-gpu=24G python Cut_paste_v1.py --root ./Cut_paste_v1/ --data_root 'carpet' --rnd_seed 2 --num_epochs 150  --dice_weight 0.5  --lr 0.0005 --batch_size 2 --lr_schedule_step_size 20 --apply_augmentations --lam_weights 1.0 1.0 1.0  &

wait
