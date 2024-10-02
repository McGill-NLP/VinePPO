local hf_model_name = 'gpt2';
local ds_config = std.prune((import '../deepspeed/zero_2.jsonnet') + {
    scheduler: {},
});

{
    trainer+: {
        type: 'ppo',
        cache_reference_model_on_temp_storage: true,

        actor_model+: {
            type: 'pretrained_causal_lm',
            hf_model_name: hf_model_name,
            pretrained_args: {
                use_flash_attention_2: false,
            },
        },
        actor_deepspeed_config: ds_config,

        critic_model+: {
            type: 'pretrained_causal_lm_with_value_head',
            value_head_dropout: 0.0,
            pretrained_backbone_model+: {
                type: 'pretrained_causal_lm',
                pretrained_args: {
                    use_flash_attention_2: false,
                },
                hf_model_name: hf_model_name,
                disable_dropout: true,
                init_base_model_only: true,
            },
        },
        critic_deepspeed_config: ds_config,

        reference_model+: {
            type: 'pretrained_causal_lm',
            pretrained_args: {
                use_flash_attention_2: false,
            },
            hf_model_name: hf_model_name,
        },
        reference_deepspeed_config: {
            gradient_accumulation_steps: 'auto',
            train_batch_size: 'auto',
            train_micro_batch_size_per_gpu: 'auto',
        },

        params+: {
            use_score_norm: false,
            use_score_scaling: false,
            adap_kl_ctrl: true,
            init_kl_coef: 0.2,
            target: 6.0,
            gamma: 1.0,
            lam: 0.95,
            cliprange: 0.2,
            cliprange_value: 0.2,
            whiten_rewards: false,
        },

        general_training_args: {
            target_train_batch_size: 128,

            per_device_train_batch_size: 128,
            per_device_eval_batch_size: 128,

            learning_rate: 1.41e-5,
            weight_decay: 0.00,
            warmup_ratio: 0.00,

            max_grad_norm: 1.0,

            dataloader_num_workers: 0,
            dataloader_pin_memory: true,

            gradient_checkpointing: false,
            torch_compile: false,
            bf16: true,

            max_seq_len: 700,

            logging_steps: 1,

            seed: std.parseInt(std.extVar('APP_SEED')),
        },

        cache_deepspeed_engines: true,
    },
}
