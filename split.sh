#python train_hier_gru_rl_split.py Tianchi-raw-full lso-test 100 100 --loss TOP1Max --hidden_act tanh \
#--user_propagation_mode init --user_to_output 0 --adapt Adagrad \
#--learning_rate 0.1 --momentum 0.1 --batch_size 100 \
#--dropout_p_hidden_usr 0.1 \
#--dropout_p_hidden_ses 0.1 \
#--dropout_p_init 0.1 \
#--n_epochs 10 \
#--eval_cutoff 1 \
#--user_key user_id --item_key item_id --session_key session_id --time_key created_at \
#--model_name Tianchi-raw-full-split-80  --strategy gate-init --rollout 5 --min_steps 4 \
#--hidden_dim 256 --state_dim 303 --action_dim 100 --min_sessions 3 --max_frames 2000000 \
#--lr_policy 1e-3 --lr_value 1e-4 --split_rl_ratio 0.8  --policy_updates 10 \
#--rnd_seed 1 --rl_model_name split80raw --reset_buffer_limit 100000

#python train_hier_gru_slices.py Tianchi-raw-full lso-test 100 100 --loss TOP1Max --hidden_act tanh \
#--user_propagation_mode init --user_to_output 0 --adapt Adagrad \
#--learning_rate 0.1 --momentum 0.1 --batch_size 100 \
#--dropout_p_hidden_usr 0.1 \
#--dropout_p_hidden_ses 0.1 \
#--dropout_p_init 0.1 \
#--n_epochs 10 \
#--eval_cutoff 1 \
#--user_key user_id --item_key item_id --session_key session_id --time_key created_at \
#--model_name Tianchi-raw-full-split-80  --strategy gate-init --rollout 5 --min_steps 4 \
#--hidden_dim 256 --state_dim 303 --action_dim 100 --min_sessions 3 --max_frames 1500000 \
#--lr_policy 1e-7 --lr_value  1e-5 --policy_updates 10 \
#--rnd_seed 1 \
#--eval_type rl-split --custom_dir rl-split80 --rl_model_name split80raw
#
python train_hier_gru_rl_split.py 30M-raw-full lso-test 100 100 --loss TOP1Max --hidden_act tanh \
--user_propagation_mode init --user_to_output 0 --adapt Adagrad --learning_rate 0.05 --momentum 0.1 \
--batch_size 100 --dropout_p_hidden_usr 0.0 --dropout_p_hidden_ses 0.1 --dropout_p_init 0.3 --n_epochs 10 \
--eval_cutoff 1 --user_key user_id --item_key item_id --session_key session_id --time_key created_at \
--model_name 30M-raw-full-split-80  --strategy gate-init --rollout 5 --min_steps 5 --hidden_dim 256 \
--state_dim 303 --action_dim 100 --min_sessions 3 --max_frames 1000000 --lr_policy 1e-7 --lr_value 1e-5 \
--split_rl_ratio 0.8 --policy_updates 10  --reset_buffer_limit 100000 --rnd_seed 1 \
--rl_model_name gateaddinit-2

python train_hier_gru_rl_split.py Tianchi-raw-full lso-test 100 100 --loss TOP1Max --hidden_act tanh \
--user_propagation_mode init --user_to_output 0 --adapt Adagrad --learning_rate 0.1 --momentum 0.1 \
--batch_size 100 --dropout_p_hidden_usr 0.0 --dropout_p_hidden_ses 0.1 --dropout_p_init 0.3 --n_epochs 10 \
--eval_cutoff 1 --user_key user_id --item_key item_id --session_key session_id --time_key created_at \
--model_name Tianchi-raw-full-split-80  --strategy gate-init --rollout 5 --min_steps 5 --hidden_dim 256 \
--state_dim 303 --action_dim 100 --min_sessions 3 --max_frames 1000000 --lr_policy 1e-4 --lr_value 1e-3 \
--split_rl_ratio 0.8 --policy_updates 10  --reset_buffer_limit 100000 --rnd_seed 1 \
--rl_model_name gateaddinit-2

python train_hier_gru_rl_split.py 30M-raw-full lso-test 100 100 --loss TOP1Max --hidden_act tanh \
--user_propagation_mode init --user_to_output 0 --adapt Adagrad --learning_rate 0.05 --momentum 0.1 \
--batch_size 100 --dropout_p_hidden_usr 0.0 --dropout_p_hidden_ses 0.1 --dropout_p_init 0.3 --n_epochs 10 \
--eval_cutoff 1 --user_key user_id --item_key item_id --session_key session_id --time_key created_at \
--model_name 30M-raw-full-split-80  --strategy add-noise-init --rollout 5 --min_steps 5 --hidden_dim 256 \
--state_dim 303 --action_dim 100 --min_sessions 3 --max_frames 1000000 --lr_policy 1e-7 --lr_value 1e-5 \
--split_rl_ratio 0.8 --policy_updates 10  --reset_buffer_limit 100000 --rnd_seed 1 \
--rl_model_name noiseaddinit-2

python train_hier_gru_rl_split.py Tianchi-raw-full lso-test 100 100 --loss TOP1Max --hidden_act tanh \
--user_propagation_mode init --user_to_output 0 --adapt Adagrad --learning_rate 0.1 --momentum 0.1 \
--batch_size 100 --dropout_p_hidden_usr 0.0 --dropout_p_hidden_ses 0.1 --dropout_p_init 0.3 --n_epochs 10 \
--eval_cutoff 1 --user_key user_id --item_key item_id --session_key session_id --time_key created_at \
--model_name Tianchi-raw-full-split-80  --strategy add-noise-init --rollout 5 --min_steps 5 --hidden_dim 256 \
--state_dim 303 --action_dim 100 --min_sessions 3 --max_frames 1000000 --lr_policy 1e-4 --lr_value 1e-3 \
--split_rl_ratio 0.8 --policy_updates 10  --reset_buffer_limit 100000 --rnd_seed 1 \
--rl_model_name noiseaddinit-2
#python train_hier_gru_slices.py 30M-raw-full lso-test 100 100 --loss TOP1Max --hidden_act tanh \
#--user_propagation_mode init --user_to_output 0 --adapt Adagrad \
#--learning_rate 0.05 --momentum 0.1 --batch_size 100 \
#--dropout_p_hidden_usr 0.1 \
#--dropout_p_hidden_ses 0.1 \
#--dropout_p_init 0.0 \
#--n_epochs 10 \
#--eval_cutoff 1 \
#--user_key user_id --item_key item_id --session_key session_id --time_key created_at \
#--model_name 30M-raw-full-split-80  --strategy gate-init --rollout 5 --min_steps 4 \
#--hidden_dim 256 --state_dim 303 --action_dim 100 --min_sessions 3 --max_frames 1000000 \
#--lr_policy 1e-7 --lr_value  1e-5  --policy_updates 10 \
#--rnd_seed 1 --eval_type rl-split --custom_dir rl-split80 --rl_model_name split80raw
