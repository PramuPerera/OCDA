
EXP_NAME='exp'
DATA_PATH='../../datasets/'
DATA_NAME='cifar32'
NET_TYPE='ecd'
METHOD='c2ae'
LEARNING_RATE=0.001
ITR=25000
NO_CLOSED=20
NO_OPEN=200
GROWTH_RATE=24
DEPTH=92

python main.py --dataset_path ${DATA_PATH} --dataset_name ${DATA_NAME} --model_mode train\
				--method ${METHOD} --experiment_name exp --lr ${LEARNING_RATE} --iterations ${ITR}\
				--no_closed ${NO_CLOSED} --no_open ${NO_OPEN} --growth_rate ${GROWTH_RATE} --depth ${DEPTH}\
				--network_type ${NET_TYPE}
