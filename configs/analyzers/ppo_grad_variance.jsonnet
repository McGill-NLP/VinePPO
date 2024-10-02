{
    type: 'ppo_gradient_variance',

    actor_deepspeed_config: {
        bf16: { enabled: true },
        wall_clock_breakdown: false,
        prescale_gradients: false,
        gradient_accumulation_steps: 'auto',
        train_batch_size: 'auto',
        train_micro_batch_size_per_gpu: $.per_device_batch_size,
    },

    max_num_checkpoints: 10,

    num_bootstrap_samples: 32,
    num_bootstrap_runs: 32,

    per_device_batch_size: 16,

    store_rolling_aggregates_on_cpu: false,
}
