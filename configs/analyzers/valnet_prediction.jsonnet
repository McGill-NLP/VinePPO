local num_expansion_rounds = 16;
local guidance_program = '{{prefix}}{{gen "chain_of_thought" temperature={temperature} top_p={top_p} max_tokens={max_tokens} save_stop_text="stop_text" stop={stop} n={num_samples}}}';

{
    type: 'valnet_prediction',

    max_num_checkpoints: 10,
    max_num_requests: 100,

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

    vllm_server+: {
        swap_space: 32,
        enable_prefix_caching: true,
    },
}
