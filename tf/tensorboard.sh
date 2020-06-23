# for viewing checkpoint progress in tensorboard

export STORAGE_BUCKET=gs://mathsreasoning
export PROBLEM=algorithmic_math_deepmind_all
declare -i PORT=2222
for folder in universal_transformer-global-2020-06-23/ universal_transformer-global-lowerlr0-02-2020-06-23/ universal_transformer-lowerlr0-02-2020-06-23/ universal_transformer-ut-lowerlr0-002-2020-06-23/
do
	export TRAIN_DIR=${STORAGE_BUCKET}/t2t_train/$PROBLEM/$folder
	tensorboard --logdir=${TRAIN_DIR}  --host localhost --port $PORT &
	PORT+=1
done
