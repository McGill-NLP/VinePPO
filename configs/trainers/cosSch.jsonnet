{
    trainer+: {
        deepspeed_config+: {
            scheduler: {
                type: 'WarmupCosineLR',
                params: {
                    last_batch_iteration: -1,
                    total_num_steps: 'auto',
                    warmup_num_steps: 'auto',
                },
            },
        },
    },
}
