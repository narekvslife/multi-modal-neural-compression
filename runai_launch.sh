wandb login ${WANDB_API_KEY}

git pull

echo hey $PRETRAINED $MODEL_NUMBER $QUALITY $TASKS $DATASET $LAMBDA $EPOCHS $LATENT_SIZE $LEARNING_RATE_MAIN $LEARNING_RATE_AUX $NUM_WORKERS $BATCH_SIZE $DEVICES

python src/train.py --pretrained $PRETRAINED --model $MODEL_NUMBER --quality $QUALITY --tasks $TASKS --dataset $DATASET --lmbda $LAMBDA --epochs $EPOCHS --latent-size $LATENT_SIZE --learning-rate-main $LEARNING_RATE_MAIN --learning-rate-aux $LEARNING_RATE_AUX --num-workers $NUM_WORKERS --batch-size $BATCH_SIZE --devices $DEVICES --accelerator $ACCELERATOR
