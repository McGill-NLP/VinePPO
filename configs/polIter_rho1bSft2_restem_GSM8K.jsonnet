local hf_model_name = 'realtreetune/rho-1b-sft-GSM8K';
local task = (import 'tasks/gsm8k_orig_format.jsonnet');
local total_num_iterations = 10;

local gsm8k_inference_pipeline =
    (import 'prompt_library/generic_GSM8K_step_by_step.jsonnet')
    + (import 'inference_strategies/tree/iid_expander.jsonnet')
    + (import 'inference_strategies/cot.jsonnet')
    + {
        inference_strategy+: {
            max_concurrent_programs: 512,
            max_concurrent_generations: 128,

            node_expander+: {
                type: 'efficient_iid',
                program_kwargs: {
                    temperature: 0.35,
                    top_p: 0.9,
                    max_tokens: 1024,
                    stop: '"\n\n\nProblem:"',
                },
                node_text_template: '{chain_of_thought}',

                // Needed to compute max_tokens on the fly
                model_context_size: 2047,
                tokenizer: {
                    type: 'pretrained',
                    hf_model_name: hf_model_name,
                },
            },
            answer_extractor+: {
                type: 'identity',
                node_key_name: 'text',
            },
            samples: 16,
            max_depth: 10,

            guidance_llm: (import 'guidance_llms/rho1b.jsonnet') + { api_base: 'none' },
            no_cache: false,
            question_field: 'query',

            seed: 42,
        },
        task: (import 'tasks/gsm8k_orig_format.jsonnet'),
        analyzers: [(import 'analyzers/task_performance.jsonnet')],
    };

local gsm8k_validation_inference_pipeline =
    gsm8k_inference_pipeline
    + {
        dataset_split: 'validation',
        dataset_portion: 1,
        inference_name: 'gsm8k_validation',
    };

(import 'polIter_rho1bSft2_restem_MATH.jsonnet')
+ {
    episode_generator+: {
        // Override the task
        task: task,
        reward_function+: { math_task: $.episode_generator.task },

        initial_model_name_or_path: hf_model_name,

        inference_strategy+: {
            guidance_llm: (import 'guidance_llms/rho1b-sft-GSM8K.jsonnet') + { api_base: 'none' },
        },
    },
    num_iterations: total_num_iterations,
    trainer +: {
        early_stop_inference_pipeline_cfg : gsm8k_validation_inference_pipeline +
        {
            inference_strategy+: {
                    node_expander+: {
                        tokenizer: $.tokenizer,
                },
            },
        },
    },
}
+ (import 'sft_rho1b_for_gsm8k_eval.jsonnet')