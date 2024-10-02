{
    episode_generator+: {
        type: 'tree_dpo_pair',
        repeat_early_stopped_paths: false,
        discard_identical_pairs: true,

        branch_factor: 3,
        max_depth: 5,
    },
}