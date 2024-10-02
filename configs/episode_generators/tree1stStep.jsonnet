{
    episode_generator+: {
        type: 'first_step_from_tree',
        repeat_early_stopped_paths: true,
        branch_factor: 3,
        max_depth: 5,
    },

    trainer+: {
        training_args+: {
            # This learning has been selected from hyperparameter search
            # on GSM8K validation set.Search space: 3e-5, 1e-5, 3e-6, 1e-6, 3e-7
            learning_rate: 3e-6,
            save_steps: 60,
            checkpoint_keep_steps: 60,
        },
    },
}