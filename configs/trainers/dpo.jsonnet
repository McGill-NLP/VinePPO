{
    trainer+: {
        type: 'dpo',

        // Taken from https://github.com/huggingface/alignment-handbook/blob/main/recipes/zephyr-7b-beta/dpo/config_full.yaml:
        // Other valid option suggested by https://arxiv.org/pdf/2311.10702.pdf
        // beta: 0.1 & num_epochs_per_iteration: 3
        dpo_beta: 0.01,
        num_epochs_per_iteration: 2,

        dpo_loss_type: 'sigmoid',
        dpo_sequence_logp_reduction: 'sum',
        dpo_use_reference_model: true,

        training_args: (import 'training_args.jsonnet') + {
            learning_rate: 5e-7,
            warmup_ratio: 0.1,

            # Total batch size for training = 128
            per_device_train_batch_size: 2,
            gradient_accumulation_steps: 8,
            gradient_checkpointing: false,

            save_steps: 150,
            checkpoint_keep_steps: 300,
        },

        deepspeed_config: (import '../deepspeed/zero_2.jsonnet')
    },

    use_deepspeed: true,
}
