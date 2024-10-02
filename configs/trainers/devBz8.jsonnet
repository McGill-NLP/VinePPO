{
    trainer+: {
        general_training_args+: {
            per_device_train_batch_size: 8,
            gradient_accumulation_steps: null,  // Will be auto computed
        },
    },
}
