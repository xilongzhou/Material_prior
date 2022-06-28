Code of Mateiral Prior:

set up environment using `environment.yml`  or build up your own environment

optimization script:

`python batch_optim2.py --load_option rand --name_pf tt --ckpt_dir folder_name --loss TD+32L1 --total_iter 2000 --scale_opt --H_inten 10 --lr 0.005 --run_option opt`

When running this, images in `data` folder will be loaded per class and results will be save to the directory like below:

 `folder_name/stone/stone_wall_2/TD+1e-132L1_3l16c5k_5in5o_tt_rand_0`

for `TD+1e-132L1_3l16c5k_5in5o_tt_rand_0`: "TD+1e-132L1" means loss; "3l16c5k" mean architecture (3 layer pixconv, 16 channel, 5 kernel size chaconv)