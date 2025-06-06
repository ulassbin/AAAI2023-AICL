# Get time here
date_str=$(date +%d_%m_%H_%M)
exp_name="Thumos_VAE_JUN6_$date_str"
python main_thumos.py \
--batch_size 75 \
--exp_name $exp_name \
--model_name ThumosModel \
--num_epochs 1200 \
--lr 0.0001 \
--detection_inf_step 50 \
--verbose  \
--soft_nms \
--data_path /abyss/home/THUMOS14
