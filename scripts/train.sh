# Get time here
date_str=$(date +%d_%m_%H_%M)
exp_name="Thumos_Combined50Sampling_$date_str"
python main_thumos.py \
--exp_name $exp_name \
--model_name ThumosModel \
--num_epochs 900 \
--detection_inf_step 50 \
--verbose  \
--soft_nms \
--data_path /abyss/home/THUMOS14
