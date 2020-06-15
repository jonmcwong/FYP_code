BEAM_SIZE=4

# register the tpu you're gonna use
export TPU_IP_ADDRESS=10.249.128.34
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

export PROBLEM=algorithmic_math_deepmind_all
export MODEL=transformer
export HPARAMS_SET=transformer_tpu

export TPU_NAME=paper-default-lowerlr	# different for each run
export STORAGE_BUCKET=gs://mathsreasoning
export MODEL_TAG=mds_paper_settings_lowerlr
export MODEL_TAG=${MODEL_TAG}-$(date +%F)

export DATA_DIR=${STORAGE_BUCKET}/t2t-data
export TMP_DIR=${STORAGE_BUCKET}/t2t_datagen
export TRAIN_DIR=${STORAGE_BUCKET}/t2t_train/$PROBLEM/$MODEL-$MODEL_TAG

export BASE_PATH=
export CHECKPOINT_SUMMARY_FILE=
for module in add_or_sub add_sub_multiple mul div mul_div_multiple mixed simplify_surd
do

# go through every line in the checkpoint file besides the first

# read the ceckpoint name
#load the checkpoint name and then collect accuracies for each of the files
#append accuracies to their own files
# load next checkpoint and repeat

t2t-decoder \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=translation.en