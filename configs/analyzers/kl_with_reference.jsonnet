// local guidance_program = '{{prefix}}{{gen "chain_of_thought" temperature={temperature} top_p={top_p} max_tokens={max_tokens} save_stop_text="stop_text" stop={stop} n={num_samples}}}';
{
    type: "kl_with_reference",

//    inference_strategy: { # it should really be read from its policy iteratino config with $.episode_generator.inference_strategy, this is just a placeholder to give you and idea of the structure, commented out so not mistaken
//        type: 'cot',
//
//        max_concurrent_programs: 16,
//        max_concurrent_generations: 16,
//
//        samples: 16,
//        max_depth: 100,
//
//        node_expander: {
//            type: 'efficient_iid',
//            program: guidance_program,
//            program_kwargs+: {
//                temperature: 0.6,
//                top_p: 0.9,
//                max_tokens: 1024,
//                stop: '"\n\n\nProblem:"',
//            },
//            node_text_template: '{chain_of_thought}',
//            num_expansion_rounds: 1,
//        },
//
//
//        answer_extractor: {
//            type: 'identity_with_solution_prefix',
//            node_key_name: 'full_text',
//            solution_prefix: '\nSolution:',
//        },
//
//        question_field: 'query',
//        question_template: '{query}',
//
//        no_cache: true,
//    },

    vllm_server+: {
        swap_space: 8,
        enable_prefix_caching: true,
    },

    actor_deepspeed_config: {
        bf16: { enabled: true },
        wall_clock_breakdown: false,
        prescale_gradients: false,
        gradient_accumulation_steps: 'auto',
        train_batch_size: 'auto',
        train_micro_batch_size_per_gpu: 'auto',
    },
}