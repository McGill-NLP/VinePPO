local ds_stage_2 = (import '../deepspeed/zero_2.jsonnet');

{
    trainer+: {
        type: 'ppo',

        actor_model+: {
            type: 'pretrained_causal_lm',
            disable_dropout: true,
            pretrained_args+: {
                use_flash_attention_2: true,
            },
        },
        actor_deepspeed_config: ds_stage_2,

        critic_model+: {
            type: 'hf_tldr_reward_model',
            disable_dropout: true,
        },
        critic_deepspeed_config: ds_stage_2,

        reference_model+: {
            type: 'pretrained_causal_lm',
            pretrained_args+: {
                use_flash_attention_2: true,
            },
        },
        reference_deepspeed_config: {
            gradient_accumulation_steps: 'auto',
            train_batch_size: 'auto',
            train_micro_batch_size_per_gpu: 'auto',
        },

        params+: {
            use_score_norm: false,
            use_score_scaling: false,

            adap_kl_ctrl: false,
            init_kl_coef: 0.05,

            gamma: 1.0,
            lam: 0.95,

            cliprange: 0.2,
            cliprange_value: 0.2,

            whiten_rewards: false,
            whiten_advantages: false,
        },

        cache_deepspeed_engines: true,
        move_reference_model_to_cpu: true,

        general_training_args: {
            target_train_batch_size: 64,

            gradient_accumulation_steps: 4,

            learning_rate: 3e-6,
            adam_epsilon: 1e-5,

            weight_decay: 0.00,
            warmup_steps: 0,

            max_grad_norm: 1.0,

            dataloader_num_workers: 1,
            dataloader_pin_memory: false,

            gradient_checkpointing: true,
            bf16: true,

            logging_steps: 1,

            seed: std.parseInt(std.extVar('APP_SEED')),
        },
    },
}
