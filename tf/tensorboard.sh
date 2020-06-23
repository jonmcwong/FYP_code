# for viewing checkpoint progress in tensorboard

export STORAGE_BUCKET=gs://mathsreasoning
export PROBLEM=algorithmic_math_deepmind_all
declare -i PORT=2222
for folder in universal_transformer-base_test-loss-0001-2020-06-21/ universal_transformer-base_test-loss-001-2020-06-21/ universal_transformer-base_test_loss_0005-2020-06-21/
do
	export TRAIN_DIR=${STORAGE_BUCKET}/t2t_train/$PROBLEM/$folder
	tensorboard --logdir=${TRAIN_DIR}  --host localhost --port $PORT &
	PORT+=1
done
