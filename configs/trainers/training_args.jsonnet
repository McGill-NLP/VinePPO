{
    per_device_train_batch_size: 32,
    per_device_eval_batch_size: 8,

    gradient_accumulation_steps: 1,

    learning_rate: 1e-5,
    weight_decay: 0.00,
    warmup_ratio: 0.06,

    max_grad_norm: 1.0,

    dataloader_num_workers: 8,
    dataloader_pin_memory: true,

    gradient_checkpointing: true,
    torch_compile: true,
    bf16: true,

    max_seq_len: 700,

    logging_steps: 1,
    save_steps: 750,

    seed: std.parseInt(std.extVar('APP_SEED'))
}
