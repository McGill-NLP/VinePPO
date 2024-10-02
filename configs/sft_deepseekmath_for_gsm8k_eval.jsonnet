local num_samples = 16;
local temperature = 0.35;

local tokenizer = {
    type: 'pretrained',
    hf_model_name: 'deepseek-ai/deepseek-math-7b-base',
};

local greedy_program = '{{prefix}}{{gen "chain_of_thought" temperature={temperature} max_tokens={max_tokens} save_stop_text="stop_text" stop={stop} n={num_samples} seed={seed}}}';

local math_inference_pipeline =
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
                    temperature: temperature,
                    top_p: 0.9,
                    max_tokens: 1024,
                    stop: '"\n\n\nProblem:"',
                },
                node_text_template: '{chain_of_thought}',

                // Needed to compute max_tokens on the fly
                model_context_size: 2047,
                tokenizer: tokenizer,
            },
            answer_extractor+: {
                type: 'identity',
                node_key_name: 'text',
            },
            samples: num_samples,
            max_depth: 10,

            guidance_llm: (import 'guidance_llms/deepseekmath7b_base.jsonnet') + { api_base: 'none' },
            no_cache: false,
            question_field: 'query',

            seed: 42,
        },
        task: (import 'tasks/gsm8k_orig_format.jsonnet'),
        analyzers: [(import 'analyzers/task_performance.jsonnet')],
    };

local gsm8k_train_inference_pipeline =
    math_inference_pipeline
    + {
        dataset_split: 'train',
        dataset_portion: 0.05253521,  // About 373 samples
        dataset_shuffle_before_portion: true,
        inference_name: 'gsm8k_train',
    };

local gsm8k_test_inference_pipeline =
    math_inference_pipeline
    + {
        dataset_split: 'test',
        dataset_portion: 1,
        inference_name: 'gsm8k_test',
    };

local gsm8k_validation_inference_pipeline =
    math_inference_pipeline
    + {
        dataset_split: 'validation',
        dataset_portion: 1,
        inference_name: 'gsm8k_validation',
    };

{
    inference_pipelines: [
        gsm8k_test_inference_pipeline,
        gsm8k_validation_inference_pipeline,
        gsm8k_train_inference_pipeline,
    ],

    evaluation_vllm_server: {},
}
