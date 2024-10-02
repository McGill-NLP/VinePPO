{
    episode_generator+: {
        type: 'tree_dpo_pair',
        only_include_max_diff: true,
        discard_identical_pairs: true,

        repeat_early_stopped_paths: false,
        branch_factor: 3,
        max_depth: 5,
    },
}