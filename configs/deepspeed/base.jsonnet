{
    optimizer: (import 'optimizer.jsonnet'),
    scheduler: (import 'lr_scheduler.jsonnet'),

    gradient_accumulation_steps: 'auto',
    gradient_clipping: 'auto',
    train_batch_size: 'auto',
    train_micro_batch_size_per_gpu: 'auto',

    zero_allow_untested_optimizer: true,

    bf16: {
        enabled: 'auto',
    },
}
