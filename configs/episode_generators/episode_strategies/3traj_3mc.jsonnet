{
    episode_generator+: {
        num_trajectories_per_tree: 3,
        num_monte_carlo_samples: 3,

        episode_strategy+: {
            type: 'tree',
            path_filters: [],
            path_aggregators: [],
            path_post_processors: [
                'fill_advantage_from_score',
            ],
        },
    },

    trainer+: {
        training_args+: {
            save_steps: 20,
            checkpoint_keep_steps: 20,
        },
    },
}
