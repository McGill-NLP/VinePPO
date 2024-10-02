{
    episode_generator+: {
        type: 'full_continuation_dpo',
        num_episodes_per_tree: 13,

        repeat_early_stopped_paths: false,
        branch_factor: 3,
        max_depth: 5,
    },
}