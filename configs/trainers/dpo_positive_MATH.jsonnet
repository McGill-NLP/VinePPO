{
    trainer+: {
        type: 'dpo_positive',

        actor_model+: {
            type: 'pretrained_causal_lm',
            disable_dropout: true,
            pretrained_args+: {
                use_flash_attention_2: true,
            },
        },

        actor_deepspeed_config: (import '../deepspeed/zero_2.jsonnet'),

        reference_model+: {
            type: 'pretrained_causal_lm',
            pretrained_args+: {
                use_flash_attention_2: true,
            },
        },

        reference_deepspeed_config: {
            bf16: { enabled: true },
            wall_clock_breakdown: false,
            prescale_gradients: false,
            gradient_accumulation_steps: 'auto',
            train_batch_size: 'auto',
            train_micro_batch_size_per_gpu: 'auto',
        },

        general_training_args: {
            target_train_batch_size: 64,

            per_device_train_batch_size: 8,

            learning_rate: 1e-6,
            weight_decay: 0.00,
            warmup_ratio: 0.03,

            max_grad_norm: 1.0,

            dataloader_num_workers: 1,
            dataloader_pin_memory: false,

            gradient_checkpointing: true,
            bf16: true,

            logging_steps: 1,

            save_steps: 32,
            checkpoint_keep_steps: 32,

            seed: std.parseInt(std.extVar('APP_SEED')),
        },

        num_epochs_per_iteration: 8,
        cache_deepspeed_engines: true,
        move_reference_model_to_cpu: true,
    },
}
