# transformer script with original mds params

# register the tpu you're gonna use
export TPU_IP_ADDRESS=10.213.119.226
#export TPU_IP_ADDRESS=10.211.245.50
#export TPU_IP_ADDRESS=10.31.105.170

export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

export PROBLEM=algorithmic_math_deepmind_all
export MODEL=neural_deque_model
export HPARAMS_SET=neural_deque

export TPU_NAME=ut-test	# different for each run
export STORAGE_BUCKET=~/Repos/final_year_at_ic/awesome_project
export MODEL_TAG=gpu-test-default-settings
export MODEL_TAG=${MODEL_TAG}-$(date +%F)

export DATA_DIR=${STORAGE_BUCKET}/mathematics_dataset-v1.0
export TMP_DIR=${STORAGE_BUCKET}/
export TRAIN_DIR=${STORAGE_BUCKET}/t2t_train/$PROBLEM/$MODEL-$MODEL_TAG

# all the correct directories already exist. no need to mkdir

# Generate data
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM

# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS_SET \
  --output_dir=$TRAIN_DIR \
  --train_steps=500000 \
  --eval_steps=3 \
  --save_checkpoints_secs=1800 \
  # --cloud_mlengine \
  # --cloud_mlengine_master_type=cloud_tpu \
  # --autotune_objective='metrics-algorithmic_math_deepmind_all/loss' \
  # --autotune_maximize=True \
  # --autotune_max_trials=100 \
  # --autotune_parallel_trials=3 \
  # --worker_gpu=0 \
  # --log_step_count_steps=500 \
  # --save_checkpoints_secs=1800
  # --hparams='batch_size=1024', \
  # --hparams='clip_grad_norm=0.1', \
  # --hparams='dropout=0.1', \
  # --hparams='label_smoothing=0', \
  # --hparams='optimizer=Adam', \
  # --hparams='learning_rate_schedule="constant"', \
  # --hparams='learning_rate_constant=6e-4', \
  # --hparams='learning_rate=6e-4', \
  # --hparams='optimizer_adam_epsilon=1e-9', \
  # --hparams='optimizer_adam_beta1=0.9', \
  # --hparams='optimizer_adam_beta2=0.995
