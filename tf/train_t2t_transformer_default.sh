# transformer script with original mds params


# register the tpu you're gonna use
export TPU_IP_ADDRESS=10.230.208.130
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

export PROBLEM=algorithmic_math_deepmind_all
export MODEL=transformer
export HPARAMS=transformer_tpu

export TPU_NAME=transformer-dropout03  # different for each run
export STORAGE_BUCKET=gs://mathsreasoning
export MODEL_TAG=base_test-dropout03
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
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --use_tpu=True \
  --cloud_tpu_name=${TPU_NAME} \
  --train_steps=1200000 \
  --eval_steps=3 \
  --save_checkpoints_secs=1800 \
  --hparams='clip_grad_norm=0.1, dropout=0.3, label_smoothing=0, optimizer=Adam, learning_rate_schedule=constant, learning_rate_constant=6e-4, learning_rate=6e-4, optimizer_adam_epsilon=1e-9, optimizer_adam_beta1=0.9, optimizer_adam_beta2=0.995'
