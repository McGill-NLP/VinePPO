{
    episode_generator+: {
        episode_strategy+: {
            type: 'tree',
            path_filters: [],
            path_aggregators: [],
            path_post_processors: [
                'negative_advantages_to_zero',
                'chop_zero_advantage_tail',
            ],
        },
    },
}
