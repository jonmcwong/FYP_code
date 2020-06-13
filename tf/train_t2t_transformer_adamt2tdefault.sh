# transformer script with t2t default adam params

export PROBLEM=algorithmic_math_deepmind_all
export MODEL=transformer
export HPARAMS=transformer_tpu
export TPU_NAME=t2t-adam-default-0	# different for each run
export STORAGE_BUCKET=gs://mathsreasoning

export DATA_DIR=${STORAGE_BUCKET}/t2t-data
export TMP_DIR=${STORAGE_BUCKET}/t2t_datagen
export TRAIN_DIR=${STORAGE_BUCKET}/t2t_train/$PROBLEM/$MODEL-$HPARAMS

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
  --hparams='optimizer=Adam' \
  --output_dir=$TRAIN_DIR \
  --use_tpu=True \
  --cloud_tpu_name=${TPU_NAME} \
  --train_steps=600000 \
  --eval_steps=3 \
  --save_checkpoints_secs=1800
