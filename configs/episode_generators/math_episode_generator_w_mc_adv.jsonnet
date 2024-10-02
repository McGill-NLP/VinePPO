(import 'math_episode_generator.jsonnet') + {
    episode_generator+: {
        type: 'math_episode_generator_w_mc_advantages',
        max_step_for_value_estimation: 20,
    },
}
