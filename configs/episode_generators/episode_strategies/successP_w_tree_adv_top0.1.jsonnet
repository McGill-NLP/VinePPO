{
    episode_generator+: {
        episode_strategy+: {
            type: 'tree',
            path_filters: [
                'successful',
            ],
            path_aggregators: [
                {
                    type: 'top_k_percent',
                    top_k_percent: 0.1,
                    keep_at_least_one: true,
                    reduction: 'mean',
                },
            ],
            path_post_processors: [
                'zero_advantages_to_epsilon',
                'negative_advantages_to_zero',
            ],
        },
    },

    trainer+: {
        num_epochs_per_iteration: 5,

        training_args+: {
            save_steps: 50,
        },
    },
}
