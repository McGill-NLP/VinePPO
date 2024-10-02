{
    episode_generator+: {
        episode_strategy+: {
            type: 'tree',
            path_filters: [
                'successful',
            ],
            path_aggregators: [
                'none',
            ],
            path_post_processors: [
                'zero_advantages_to_epsilon',
                'negative_advantages_to_zero',
            ],
        },
    },
}
