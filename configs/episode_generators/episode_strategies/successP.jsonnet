{
    episode_generator+: {
        episode_strategy+: {
            type: 'tree',
            path_filters: [
                'successful',
            ],
            path_aggregators: [],
            path_post_processors: [
                'all_advantages_to_one',
            ],
        },
    },
}
