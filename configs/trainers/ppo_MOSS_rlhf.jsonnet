local ds_stage_2_w_cpu_optimizer = (import '../deepspeed/zero_2.jsonnet') + {
    zero_optimization+: {
        offload_optimizer+: {
            device: 'cpu',
            pin_memory: true,
        },
    },

    memory_breakdown: true,
};

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
        actor_deepspeed_config: ds_stage_2_w_cpu_optimizer,

        critic_model+: {
            type: 'moss_llama_reward_model',
            disable_dropout: true,
            pretrained_args: {
                use_flash_attention_2: true,
            },
        },
        critic_deepspeed_config: ds_stage_2_w_cpu_optimizer + {
            optimizer+: {
                params: { lr: 1.5e-6 },
            },
        },

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
            use_score_norm: true,
            use_score_scaling: true,

            adap_kl_ctrl: false,
            init_kl_coef: 0.01,

            gamma: 1.0,
            lam: 0.95,

            cliprange: 0.2,
            cliprange_value: 0.2,

            whiten_rewards: false,
            whiten_advantages: false,
        },

        general_training_args: {
            target_train_batch_size: 32,

            per_device_train_batch_size: 8,

            learning_rate: 5e-7,
            weight_decay: 0.00,
            warmup_steps: 100,

            max_grad_norm: 1.0,

            dataloader_num_workers: 4,
            dataloader_pin_memory: true,

            gradient_checkpointing: true,
            bf16: true,

            logging_steps: 1,

            seed: std.parseInt(std.extVar('APP_SEED')),
        },
    },
}
