Code of Mateiral Prior:

set up environment using `environment.yml`  or build up your own environment

To run using our provided data and pattern, run this script:

`python batch_optim2.py --load_option rand --name_pf $filename --ckpt_dir $foldername --loss TD+32L1 --total_iter 2000 --scale_opt --H_inten 10 --lr 0.005 --run_option opt`

When running this, images in `data` folder will be loaded per class, $filename and $foldername are the saved file and folder path, results will be save to the directory like below:

 `$foldername/stone/stone_wall_2/TD+1e-132L1_3l16c5k_5in5o_$filename_rand_0`

for `TD+1e-132L1_3l16c5k_5in5o_tt_rand_0`: "TD+1e-132L1" means loss; "3l16c5k" mean architecture (3 layer pixconv, 16 channel, 5 kernel size chaconv)

To run on your customized data, use this script:

`python cust_optim.py --name_pf $filename --ckpt_dir $foldername --loss TD+32L1 --total_iter 2000 --scale_opt --H_inten 10 --lr 0.005 --run_option opt --in_img_path $imgpath --in_pat_path $patpath`

where $imgpath and $patpath are the specified path of image and patterns, all the patterns in this directory will be used as input to the network
