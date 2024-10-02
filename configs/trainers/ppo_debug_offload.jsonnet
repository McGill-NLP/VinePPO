local hf_model_name = 'realtreetune/MOSS-sft-7b';

local deepspeed_z2_offOpt_cfg = (import '../deepspeed/zero_2.jsonnet') + {
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
        num_epochs_per_iteration: 4,

        actor_model+: {
            type: 'pretrained_causal_lm',
            hf_model_name: hf_model_name,
            pretrained_args: {
                use_flash_attention_2: true,
            },
        },
        actor_deepspeed_config: deepspeed_z2_offOpt_cfg,

        critic_model+: {
            type: 'pretrained_causal_lm_with_value_head',
            value_head_dropout: 0.0,
            pretrained_backbone_model+: {
                type: 'pretrained_causal_lm',
                pretrained_args: {
                    use_flash_attention_2: true,
                },
                hf_model_name: hf_model_name,
                disable_dropout: true,
                init_base_model_only: true,
            },
        },
        critic_deepspeed_config: deepspeed_z2_offOpt_cfg,

        reference_model+: {
            type: 'pretrained_causal_lm',
            pretrained_args: {
                use_flash_attention_2: true,
            },
            hf_model_name: hf_model_name,
        },
        reference_deepspeed_config: {
            gradient_accumulation_steps: 'auto',
            train_batch_size: 'auto',
            train_micro_batch_size_per_gpu: 'auto',
        },

        params+: {
            use_score_norm: true,
            use_score_scaling: true,
        },

        general_training_args: {
            per_device_train_batch_size: 2,
            per_device_eval_batch_size: 2,

            gradient_accumulation_steps: 1,

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
            save_steps: 750,

            seed: std.parseInt(std.extVar('APP_SEED')),
        },
    },
}
