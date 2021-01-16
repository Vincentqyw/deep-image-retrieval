#
# created by vincentqin 2021.1.16
# export match pairs using DIR which can be used in SFM feature matching.
# 

# setting paths

export DIR_ROOT=$PWD
export DB_ROOT=/PATH/TO/YOUR/DATASETS

workspace_path=$PWD

dataset_name="scene1"
input_datasets='ImageList("outputs/scene1.txt")'
images_path=$DB_ROOT

topN=50

###################################################################
# DO NOT EDIT THE FOLLOWING LINES IF YOU KNOW HOW TO DO WITH IT.
###################################################################

if [ ! -d $PWD/pairs ]; then
    mkdir -p $PWD/pairs
fi

if [ ! -d $PWD/outputs ]; then
    mkdir -p $PWD/outputs
fi

# generate images list
images_list=$workspace_path/outputs/$dataset_name.txt # DON'T EDIT!!!
python get_images_list.py --input $images_path --outputs $images_list

# extract features
db_desc_path=$workspace_path/outputs/$dataset_name.npy

if [ ! -f $db_desc_path ]; then
    python -m dirtorch.extract_features --dataset ${input_datasets} \
            --checkpoint dirtorch/data/Resnet101-AP-GeM.pt \
            --output $db_desc_path \
            --whiten Landmarks_clean --whitenp 0.25 --gpu 0
fi            

# export pairs
pairs_file_path=$PWD/pairs/pairs-db-dir$topN.txt

python -m dirtorch.test_custom.py --dataset ${input_datasets} \
		--checkpoint dirtorch/data/Resnet101-AP-GeM.pt \
		--whiten Landmarks_clean --whitenp 0.25 --gpu 0 \
        --load-feats $db_desc_path \
        --out-json $PWD/outputs/${dataset_name}_query.json \
        --images_path $images_path \
        --output_pairs $pairs_file_path \
        --topN $topN