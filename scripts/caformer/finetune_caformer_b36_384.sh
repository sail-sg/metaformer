DATA_PATH=/path/to/imagenet
CODE_PATH=/path/to/code/metaformer # modify code path here
INIT_CKPT=/path/to/trained/model

ALL_BATCH_SIZE=1024
NUM_GPU=8
GRAD_ACCUM_STEPS=4 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS


cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model caformer_b36_384 --epochs 30 --opt lamb --lr 1e-4 --sched None \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--initial-checkpoint $INIT_CKPT \
--mixup 0 --cutmix 0 \
--model-ema --model-ema-decay 0.9999 \
--drop-path 0.8 --head-dropout 0.5