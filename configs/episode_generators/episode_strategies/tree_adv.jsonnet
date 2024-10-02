{
    episode_generator+: {
        episode_strategy+: {
            type: 'tree',
            path_filters: [],
            path_aggregators: [],
            path_post_processors: [],
        },
    },
    trainer+: {
        training_args+: {
            save_steps: 750,
        }
    },
    evaluate_every_n_checkpoints: 5,
}
