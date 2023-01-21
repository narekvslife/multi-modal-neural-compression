wandb login ${WANDB_API_KEY}

git pull

python src/train.py --dataset ${DATASET} --tasks ${TASKS} --model ${MODEL_NUMBER} --pretrained ${PRETRAINED}  --quality ${QUALITY} --lmbda ${LAMBDA} --epochs ${EPOCHS} --latent-size ${LATENT_SIZE} --learning-rate-main ${LEARNING_RATE_MAIN} --learning-rate-aux ${LEARNING_RATE_AUX} --num-workers ${NUM_WORKERS} --batch-size ${BATCH_SIZE} --devices ${DEVICES} --accelerator ${ACCELERATOR}
