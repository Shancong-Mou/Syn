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

srun --exclusive -n 1 -N 1 --mem-per-gpu=24G python Cut_paste_v2_optimzing_the_proportion.py --root ./Defect_aug_with_default_policy_add_clean_only_parameter_MVTech_start30/ --data_root 'carpet' --rnd_seed 0 --num_epochs 120  --dice_weight 0.5  --lr 0.00025 --batch_size 2 --lr_schedule_step_size 20 --apply_augmentations --optimize_weights --paste_per_cat --warm_up 0 --lam_weights 1.0 1.0 1.0 1.0 1.0 1.0 | tee res_start30_0.txt &
wait
