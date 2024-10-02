local num_samples = 16;
local temperature = 0.35;

local tokenizer = {
    type: 'pretrained',
    hf_model_name: 'deepseek-ai/deepseek-math-7b-base',
};


local math_inference_pipeline =
    (import 'prompt_library/generic_MATH_step_by_step.jsonnet')
    + (import 'inference_strategies/tree/iid_expander.jsonnet')
    + (import 'inference_strategies/cot.jsonnet')
    + {
        inference_strategy+: {
            max_concurrent_programs: 512,
            max_concurrent_generations: 16,

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
                model_context_size: 4095,
                tokenizer: tokenizer,
            },
            answer_extractor+: {
                type: 'identity',
                node_key_name: 'text',
            },
            samples: num_samples,
            max_depth: 10,

            guidance_llm: (import 'guidance_llms/deepseekmath7b_base.jsonnet') + { api_base: 'none' },
            no_cache: true,
            question_field: 'query',
        },
        task: (import 'tasks/math_inplace_no_answer_prefix.jsonnet'),
        analyzers: [(import 'analyzers/task_performance.jsonnet')],

        seed: 42,
    };

local math_train_inference_pipeline =
    math_inference_pipeline
    + {
        dataset_split: 'train',
        dataset_portion: 0.04347826,  // About 500 samples
        dataset_shuffle_before_portion: true,
        inference_name: 'math_train',
    };

local math_test_inference_pipeline =
    math_inference_pipeline
    + {
        dataset_split: 'test',
        dataset_portion: 1,
        inference_name: 'math_test',
    };

local math_validation_inference_pipeline =
    math_inference_pipeline
    + {
        dataset_split: 'validation',
        dataset_portion: 1,
        inference_name: 'math_validation',
    };

local collegeMath_inference_pipeline =
    math_inference_pipeline
    + {
        task: (import 'tasks/collegeMath_inplace_no_answer_prefix.jsonnet'),
    };

local collegeMath_test_inference_pipeline =
    collegeMath_inference_pipeline
    + {
        dataset_split: 'test',
        dataset_portion: 0.1774308,
        inference_name: 'collegeMath_test',
    };

local olympiadbench_inference_pipeline =
    math_inference_pipeline
    + {
        task: (import 'tasks/olympiadbench_inplace_no_answer_prefix.jsonnet'),
    };

local olympiadbench_test_inference_pipeline =
    olympiadbench_inference_pipeline
    + {
        dataset_split: 'test',
        dataset_portion: 1,
        inference_name: 'olympiadbench_test',
    };

{
    inference_pipelines: [
        math_test_inference_pipeline,
        math_validation_inference_pipeline,
        math_train_inference_pipeline,
        collegeMath_test_inference_pipeline,
        olympiadbench_test_inference_pipeline,
    ],

    evaluation_vllm_server: {},
}
