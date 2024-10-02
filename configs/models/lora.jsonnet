{
    trainer+: {
        training_args+: {
            per_device_train_batch_size: 8,
            gradient_accumulation_steps: 16,

            # For god knows what reason, neither torch_compile nor gradient_checkpointing
            # work with LoRA. This is why the batch size is so small.
            torch_compile: false,
            gradient_checkpointing: false,

            learning_rate: 1e-4,
            weight_decay: 0.01,
        },
    },
    use_deepspeed: false,

    model+: {
        lora_config+: {
            r: 64,
            lora_alpha: 128,
            target_modules: [
                'q_proj',
                'k_proj',
                'v_proj',
                'o_proj',
                'gate_proj',
                'down_proj',
                'up_proj',
            ],
        },
    },
}
