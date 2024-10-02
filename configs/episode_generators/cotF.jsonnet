{
    episode_generator+: {
        type: 'tree',
        repeat_early_stopped_paths: false,
    },

    trainer+: {
        training_args+: {
            # This learning has been selected from hyperparameter search
            # on GSM8K validation set.Search space: 3e-5, 1e-5, 3e-6, 1e-6, 3e-7
            learning_rate: 3e-6,
        },
    },
}
