# Get time here
date_str=$(date +%d_%m_%H_%M)
exp_name="Animal_kingdom_$date_str"
python main_thumos.py \
--exp_name $exp_name \
--model_name ThumosModel \
--num_epochs 900 \
--detection_inf_step 20 \
--verbose  \
--soft_nms \
--data_path /abyss/home/THUMOS14
