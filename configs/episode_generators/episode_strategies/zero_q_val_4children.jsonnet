{
    episode_generator+: {
        episode_strategy+: {
            type: 'tree',
            path_filters: [
                'zero_last_step_score',
            ],
            path_aggregators: [],
            path_post_processors+: [
                'fill_advantage_from_score',
                'last_step_advantage_to_one',
                'apply_importance_weights',
            ],
        },
        num_first_level_children: 4,
    },


    trainer+: {
        training_args+: {
            // This learning has been selected from hyperparameter search
            // on GSM8K validation set.Search space: 3e-5, 1e-5, 3e-6, 1e-6, 3e-7
            learning_rate: 3e-6,

            save_steps: 50,
            checkpoint_keep_steps: 50,
        },
    },
}
