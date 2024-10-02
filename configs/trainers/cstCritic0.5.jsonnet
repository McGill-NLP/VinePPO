{
    trainer+: {
        critic_model: {
            type: 'always_constant_value_model',
            constant_value: 0.5,
        },
        critic_deepspeed_config: {
            bf16: { enabled: true },
            wall_clock_breakdown: false,
            prescale_gradients: false,
            gradient_accumulation_steps: 'auto',
            train_batch_size: 'auto',
            train_micro_batch_size_per_gpu: 'auto',
        },
        disable_critic_training: true,
    },
}
