{
    episode_generator+: {
        type: 'tree_state_action',

        repeat_early_stopped_paths: false,
        include_importance_weights: true,

        branch_factor: 3,
        max_depth: 5,

        episode_strategy+: {
            type: 'tree',
            path_post_processors+: [
                'non_last_nodes_score_to_zero',
                'non_last_nodes_advantage_to_zero',
            ],
        },
    },

    trainer+: {
        training_args+: {
            // This learning has been selected from hyperparameter search
            // on GSM8K validation set.Search space: 3e-5, 1e-5, 3e-6, 1e-6, 3e-7
            learning_rate: 3e-6,


            save_steps: 100,
            checkpoint_keep_steps: 100,
            gradient_accumulation_steps: 16,
        },
    },
}
