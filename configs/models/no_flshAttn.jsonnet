{
    trainer+: {
        training_args+: {
            per_device_train_batch_size: 2,
            gradient_accumulation_steps: 64,
        },
    },
    model+: {
        pretrained_args+: {
            use_flash_attention_2: false,
        },
    },
}
