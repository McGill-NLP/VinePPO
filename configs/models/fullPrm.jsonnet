{
    trainer+: {
        training_args+:{
            per_device_train_batch_size: 4,
            gradient_accumulation_steps: 8,
            gradient_checkpointing: false,
        },
        deepspeed_config: (import '../deepspeed/zero_2.jsonnet'),
    },
    use_deepspeed: true,
}
