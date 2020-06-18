# transformer script with original mds params

# register the tpu you're gonna use
export TPU_IP_ADDRESS=10.247.45.74
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

export PROBLEM=algorithmic_math_deepmind_all
export MODEL=universal_transformer
export HPARAMS_SET=adaptive_universal_transformer_global_base_tpu
export RANGED_HPARAMS=adaptive_universal_transformer_base_range_0

export TPU_NAME=actualmathstpu	# different for each run
export STORAGE_BUCKET=gs://mathsreasoning
export MODEL_TAG=range
export MODEL_TAG=${MODEL_TAG}-$(date +%F)

export DATA_DIR=${STORAGE_BUCKET}/t2t-data
export TMP_DIR=${STORAGE_BUCKET}/t2t_datagen
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
  --hparams_range=${RANGED_HPARAMS} \
  --cloud_mlengine \
  --cloud_mlengine_master_type=cloud_tpu \
  --autotune_objective='metrics-algorithmic_math_deepmind_all/accuracy' \
  --autotune_maximize \
  --autotune_max_trials=100 \
  --autotune_parallel_trials=3 \
  --worker_gpu=0

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
