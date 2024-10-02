local num_mc_rollouts = 9;

(import 'polIter_deepseekSft2_ppo_MATH.jsonnet')
+ (import 'trainers/no_critic.jsonnet')
+ {
    episode_generator+: {
        type: 'math_episode_generator_w_mc_advantages',

        vllm_server+: {
            swap_space: 32,
            enable_prefix_caching: true,
        },

        value_estimation_inference_strategy+: {
            type: 'cot',

            max_concurrent_programs: 512,
            max_concurrent_generations: 512,

            samples: num_mc_rollouts,
            max_depth: 100,  // Deprecated parameter. Doesn't do anything.

            node_expander: $.episode_generator.inference_strategy.node_expander + {
                model_context_size: $.episode_generator.max_sequence_length,
            },
            answer_extractor: {
                type: 'identity_with_solution_prefix',
                node_key_name: 'full_text',
                solution_prefix: '\nSolution:',
            },

            guidance_llm: $.episode_generator.inference_strategy.guidance_llm,

            question_field: 'query',
            question_template: '{query}',

            no_cache: true,
        },
    },

    trainer+: {
        actor_deepspeed_config: (import 'deepspeed/zero_2.jsonnet'),

        general_training_args+: {
            save_steps: 10,
            checkpoint_keep_steps: 60,
        },
    },

    analyzers: [
        (import 'analyzers/mc_value_prediction.jsonnet') + {
            task: $.episode_generator.task,
            tokenizer: $.tokenizer,
            vllm_server+: { swap_space: 64 },

            reward_function: $.episode_generator.reward_function,

            max_num_requests: 512,

            inference_strategy+: {
                guidance_llm: $.episode_generator.inference_strategy.guidance_llm,

                max_concurrent_programs: 32,
                max_concurrent_generations: 16,

                node_expander+: {
                    program_kwargs+: { temperature: $.episode_generator.inference_strategy.node_expander.program_kwargs.temperature },
                    model_context_size: $.episode_generator.max_sequence_length,
                    tokenizer: $.tokenizer,
                },
            },
        },
        (import 'analyzers/ppo_grad_variance.jsonnet') + {
            per_device_batch_size: $.trainer.general_training_args.per_device_train_batch_size,
        },
        (import 'analyzers/mc_advantage_distribution.jsonnet') + {
            max_num_iterations: 10,
        },

        (import 'analyzers/mc_value_action_ranking.jsonnet') + {
            task: $.episode_generator.task,
            tokenizer: $.tokenizer,
            vllm_server+: { swap_space: 64 },

            reward_function: $.episode_generator.reward_function,

            max_num_requests: 512,
            max_num_states: 128,

            num_mc_rollouts: $.episode_generator.value_estimation_inference_strategy.samples,

            inference_strategy+: {
                guidance_llm: $.episode_generator.inference_strategy.guidance_llm,

                max_concurrent_programs: 32,
                max_concurrent_generations: 16,

                node_expander+: {
                    program_kwargs+: { temperature: $.episode_generator.inference_strategy.node_expander.program_kwargs.temperature },
                    model_context_size: $.episode_generator.max_sequence_length,
                    tokenizer: $.tokenizer,
                },
            },
        },
    ],
}
+ (import 'episode_generators/9rolls.jsonnet')
+ (import 'trainers/refKl0.0001.jsonnet')
+ (import 'trainers/klLoss.jsonnet')