.
# for viewing checkpoint progress in tensorboard

export STORAGE_BUCKET=gs://mathsreasoning
export MODEL=universal_transformer
export MODEL_TAG=range-2020-06-18
export PROBLEM=algorithmic_math_deepmind_all

export TRAIN_DIR=${STORAGE_BUCKET}/t2t_train/$PROBLEM/$MODEL-$MODEL_TAG
tensorboard --logdir=${TRAIN_DIR}
