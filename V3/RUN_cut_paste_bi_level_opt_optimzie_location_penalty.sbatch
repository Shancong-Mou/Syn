#!/bin/bash
#SBATCH -JWhole_image_no_aug            # Job name
#SBATCH -Agts-jshi33                             # Charge account
#SBATCH --mem=50G                # RAM per node
#SBATCH --threads-per-core=1    # do not use hyperthreads (i.e. CPUs = physical cores below)
#SBATCH --cpus-per-task=2       # number of CPUs per process

## nodes allocation
#SBATCH --nodes=1               # number of nodes
#SBATCH --gres=gpu:RTX_6000:1   # number of GPUs per node (gres=gpu:N)

#SBATCH -t3600                                    # Duration of the job (Ex: 15 mins)
#SBATCH -qinferno                                # QOS name
#SBATCH -oReport-%j.out                          # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL               # Mail preferences
#SBATCH --mail-user=smou7@gatech.edu            # e-mail address for notifications

cd $HOME/scratch/Learn_augmentation            # Change to working directory created in $HOME
module load anaconda3/2022.05                   # Load module dependencies
source activate SynData

#####################150 epochs #######################



# random location baseline (Cut&Paste - Random locations )
srun --exclusive -n 1 -N 1 --mem-per-gpu=24G python Defect_aug_with_default_policy_add_clean_augmentation_proportion_only_optimize_aug_policy_add_each_aug_MV_tech_to_background_optimize_location_baseline.py --root ./Defect_aug_with_default_policy_add_clean_only_parameter_MVTech_start80_capsule_random_location/ --data_root 'capsule' --rnd_seed 0 --num_epochs 150  --dice_weight 0.5  --lr 0.00025 --batch_size 2 --lr_schedule_step_size 20 --apply_augmentations --optimize_weights --paste_per_cat --warm_up 150 --lam_weights 1.0 1.0  --train_from_scrach --save_all_images | tee res_start30_random_location_baseline_0.txt &
wait

# this is the object location baseline (Groundtruth product location )
srun --exclusive -n 1 -N 1 --mem-per-gpu=24G python Defect_aug_with_default_policy_add_clean_augmentation_proportion_only_optimize_aug_policy_add_each_aug_MV_tech_to_background_optimize_location_on_object.py --root ./Defect_aug_with_default_policy_add_clean_only_parameter_MVTech_start80_capsule_object_location/ --data_root 'capsule' --rnd_seed 0 --num_epochs 150  --dice_weight 0.5  --lr 0.00025 --batch_size 2 --lr_schedule_step_size 20 --apply_augmentations --optimize_weights --paste_per_cat --warm_up 150 --lam_weights 1.0 1.0   --save_all_images  | tee res_start30_object_location_baseline_0.txt &
wait

# optimziation location (Cut&Paste - Learned locations )
srun --exclusive -n 1 -N 1 --mem-per-gpu=24G python learn_loc_equvilent_formulation_revise_grad_non_accum_small_grad_AISTATS_pnty_1e-5_concentrate_06_capsule.py | tee res_aug_loc_start30_non_accum_small_grad_AISTATS_pnty_1e-5_concentrate_06_capsule.txt  &
wait
