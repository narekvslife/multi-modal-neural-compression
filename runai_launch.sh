git pull

python src/train.py --dataset ${DATASET} --tasks ${TASKS} --model ${MODEL_NUMBER} --latent-channels ${LATENT_CHANNELS} --conv-channels ${CONV_CHANNELS} --wandb-run-name ${WANDB_RUN_NAME} --lmbda ${LAMBDA} --epochs ${EPOCHS} --pretrained ${PRETRAINED} --quality ${QUALITY} --learning-rate-main ${LEARNING_RATE_MAIN} --learning-rate-aux ${LEARNING_RATE_AUX} --num-workers ${NUM_WORKERS} --batch-size ${BATCH_SIZE} --devices ${DEVICES} --accelerator ${ACCELERATOR}
