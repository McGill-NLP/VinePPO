local num_expansion_rounds = 16;
local guidance_program = '{{prefix}}{{gen "chain_of_thought" temperature={temperature} top_p={top_p} max_tokens={max_tokens} save_stop_text="stop_text" stop={stop} n={num_samples}}}';
local num_alternative_actions = 5;

{
    type: 'valnet_action_ranking',

    max_num_checkpoints: 10,
    max_num_requests: 100,

    min_num_alternative_actions: 3,
    max_num_states: 256,

    inference_strategy: {
        type: 'cot',

        max_concurrent_programs: 16,
        max_concurrent_generations: 16,

        samples: 256 / num_expansion_rounds,
        max_depth: 100,

        node_expander: {
            type: 'efficient_iid',
            program: guidance_program,
            program_kwargs+: {
                temperature: 1,
                top_p: 0.9,
                max_tokens: 1024,
                stop: '"\n\n\nProblem:"',
            },
            node_text_template: '{chain_of_thought}',
            num_expansion_rounds: num_expansion_rounds,
        },


        answer_extractor: {
            type: 'identity_with_solution_prefix',
            node_key_name: 'full_text',
            solution_prefix: '\nSolution:',
        },

        question_field: 'query',
        question_template: '{query}',

        no_cache: true,
    },

    alternative_continuation_inference_strategy+: $.inference_strategy + {
        samples: num_alternative_actions,
        node_expander+: {
            program_kwargs+: {
                temperature: 1,
            },
            num_expansion_rounds: 1,
        },
    },

    critic_deepspeed_config: {
        bf16: { enabled: true },
        wall_clock_breakdown: false,
        prescale_gradients: false,
        gradient_accumulation_steps: 1,
        train_batch_size: 'auto',
        train_micro_batch_size_per_gpu: $.per_device_batch_size,
    },

    per_device_batch_size: 16,

    append_bos_to_query: true,

    vllm_server+: {
        swap_space: 32,
        enable_prefix_caching: true,
    },
}
