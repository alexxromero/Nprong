#!/usr/bin/env bash

# ----- baseline networks ----- #
#python pfn_keras.py --save_dir pfn_fullrun --tag pfn
#python nsubs_mass.py --save_dir nsubs_mass_fullrun --tag nsubs_mass
#python nsubs_mass_EFPsafe.py --save_dir nsubs_mass_EFP_fullrun --tag nsubs_mass_EFP

# ----- selecting features using lasso ----- #
# strength 5 had the best performance trade-off with fewer no. of features #
#python lasso_HL.py --save_dir lasso_all --tag strength_10 --strength 10
#python lasso_HL.py --save_dir lasso_all --tag strength_5 --strength 5
#python lasso_HL.py --save_dir lasso_all --tag strength_2 --strength 2
#python lasso_HL.py --save_dir lasso_all --tag strength_1 --strength 1

# ----- and analyze with only the lasso-selected features ----- #
#python nsubs_mass_EFPsafe_selected.py --save_dir nsubs_mass_EFP_selected_fullrun --tag selected_lasso_strength5

# ----- ranking of the features ----- #
# ----- first, of the nsubs + mass ----- #
# for sort_feat in {0..135}
# do
#   python ranked_nsubs_mass.py --save_dir ranked_nsubs_mass --tag ranked_nsubs_mass_$sort_feat --sort_feat $sort_feat
# done

# # ----- and also the 31 lasso-selected features ----- #
# for sort_feat in {0..31}
# do
#   python ranked_nsubs_mass_EFPsafe_selected.py --save_dir ranked_nsubs_mass_EFPsafe_selected --tag ranked_nsubs_mass_EFP_sel_$sort_feat --sort_feat $sort_feat
# done




# -- Now, see if maybe less Nsubs are enough -- #
#python nsubs_mass.py --save_dir nsubs_mass_k15_fullrun --tag nsubs_mass --k 15
#python nsubs_mass.py --save_dir nsubs_mass_k25_fullrun --tag nsubs_mass --k 25
#python nsubs_mass.py --save_dir nsubs_mass_k35_fullrun --tag nsubs_mass --k 35

# trying with k=25, let's see how it does once we combine it with the EFPs
# It ended up doing as well as with k=45, so let's go with this one.
#python nsubs_mass_EFPsafe.py --save_dir nsubs_mass_EFP_k25_fullrun --tag nsubs_mass_EFP_k25 --k 25
# next step is to rank the features
# starting at 30 because that's where it failed the first time. Forgot to hup
# for sort_feat in {30..76}
# do
#   python ranked_nsubs_mass.py --save_dir ranked_nsubs_mass_k25 --tag ranked_nsubs_mass_k25_$sort_feat --k 25 --sort_feat $sort_feat
# done

# ted lasso
# python lasso_HL.py --save_dir lasso_k25_all --tag strength_k25_10 --strength 10 --k 25
# python lasso_HL.py --save_dir lasso_k25_all --tag strength_k25_5 --strength 5 --k 25
