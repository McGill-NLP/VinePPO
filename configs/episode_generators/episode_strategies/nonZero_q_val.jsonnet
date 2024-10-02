{
    episode_generator+: {
        type: 'tree_state_action',
        episode_strategy+: {
            type: 'tree',
            path_filters: [
                'non_zero_last_step_score',
            ],
            path_aggregators: [],
            path_post_processors+: [
                'fill_advantage_from_score',
                'apply_importance_weights',
            ],
        },
    },
}
