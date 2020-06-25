# transformer script with original mds params


# register the tpu you're gonna use
export TPU_IP_ADDRESS=10.55.236.170
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

export PROBLEM=algorithmic_math_deepmind_all
export MODEL=transformer
export HPARAMS=transformer_tpu

export TPU_NAME=base-relu-dp00	  # different for each run
export STORAGE_BUCKET=gs://mathsreasoning
export MODEL_TAG=base-relu-dp-00
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
  --train_steps=800000 \
  --eval_steps=3 \
  --save_checkpoints_secs=1800 \
  --hparams='relu_dropout=0,layer_prepostprocess_dropout=0'
