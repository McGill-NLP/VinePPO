{
    trainer+: {
        training_args+: {
            per_device_train_batch_size: 64,
        },
        deepspeed_config: (import '../deepspeed/zero_2.jsonnet'),
    },
    use_deepspeed: true,

    model+: {
        freeze_config+:{
            freeze_first_k_layers: 16,
            freeze_embeddings: true,
        }
    }
}
