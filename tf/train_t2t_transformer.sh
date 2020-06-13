
PROBLEM=algorithmic_math_deepmind_all
MODEL=transformer
HPARAMS=transformer_tpu_bf16_activation

DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM
echo "FINISHED DOWNLOADING"
# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --use_tpu-True \
  --cloud_tpu_name=${TPU_NAME} \
  --train_steps=10 \
  --eval_steps=3


