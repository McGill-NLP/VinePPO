{
    trainer+: {
        type: 'value_network',
        num_epochs_per_iteration: 10,

        training_args: (import 'training_args.jsonnet') + {
            learning_rate: 5e-7,
            warmup_ratio: 0.1,

            per_device_train_batch_size: 4,
            gradient_accumulation_steps: 8,
            gradient_checkpointing: false,

            save_steps: 150,
            checkpoint_keep_steps: 300,
        },

        deepspeed_config: (import '../deepspeed/zero_2.jsonnet')
    },

    use_deepspeed: true,
}
