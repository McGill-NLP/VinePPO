{
    trainer+: {
        num_epochs_per_iteration: 10,

        training_args+: {
            save_steps: 120,
            checkpoint_keep_steps: 240,
        },
    },
}
