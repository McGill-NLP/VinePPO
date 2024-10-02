{
    trainer+: {
        deepspeed_config+: {
            scheduler+: {
                type: 'WarmupLR',
                params: {
                    last_batch_iteration: -1,
                    warmup_min_lr: 'auto',
                    warmup_max_lr: 'auto',
                    warmup_num_steps: 'auto',
                },
            },
        },
    },
}
