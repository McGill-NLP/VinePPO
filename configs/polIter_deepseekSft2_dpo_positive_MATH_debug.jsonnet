local hf_model_name = 'realtreetune/deepseekmath-7b-sft-MATH-v2';

local math_task = (import 'tasks/math_inplace_no_answer_prefix.jsonnet') + {
    prepend_in_context_few_shot: false,
    ensure_fit_in_context_size: false,
};

local num_rollouts_per_sample = 8;
local total_num_iterations = 1;
local sampling_temperature = 0.6;

(import 'gvar.jsonnet')
+ (import 'prompt_library/MATH_step_by_step_sft.jsonnet')
+ (import 'runtimes/policy_iteration.jsonnet')
+ (import 'episode_generators/math_episode_generator.jsonnet')
+ (import 'trainers/dpo_positive_MATH.jsonnet')
+ {
    episode_generator+: {
        vllm_server+: {
            swap_space: 8,
        },
        // Override the task
        type: "math_dpo_positive_episode_generator",
        reward_threshold: 0.5,
        max_pairs_per_question: 8,
        task: math_task,
        reward_function+: { math_task: $.episode_generator.task },
        reasoning_step_delimiter: '',
        answer_prefix: null,

        initial_model_name_or_path: hf_model_name,

        dataset_sample_with_replacement: false,
        dataset_portion: 0.01,
        total_num_iterations: total_num_iterations,

        max_sequence_length: 2499,
        save_generations_every_n_iteration: 50,

        inference_strategy: {
            type: 'cot',

            max_concurrent_programs: 128,
            max_concurrent_generations: 64,

            samples: num_rollouts_per_sample,
            max_depth: 100,  // Deprecated parameter. Doesn't do anything.

            node_expander: {
                type: 'efficient_iid',
                program: $.prompt_library.tree.expansion.iid,
                program_kwargs+: {
                    temperature: sampling_temperature,
                    top_p: 0.9,
                    max_tokens: 1024,
                    stop: '"\n\n\nProblem:"',
                },
                node_text_template: '{chain_of_thought}',

                // Needed to compute max_tokens on the fly
                model_context_size: 4095,
                tokenizer: $.tokenizer,
            },

            answer_extractor: {
                type: 'identity',
                node_key_name: 'text',
            },

            guidance_llm: (import 'guidance_llms/deepseekmath7b-sft-MATH-v2.jsonnet') + { api_base: 'none' },

            question_field: 'query',
            question_template: $.prompt_library.tree.question_template,

            no_cache: false,
        },
    },

    tokenizer: {
        type: 'pretrained',
        hf_model_name: $.episode_generator.initial_model_name_or_path,
    },
    use_deepspeed: true,

    num_iterations: total_num_iterations,
    num_episodes_per_iteration: null,
    episodes_cloud_log_steps: 50,

    trainer+: {
        type: "dpo_positive",

        dpo_label_smoothing: 0.0,
        dpo_beta: 0.1,
        dpo_positive_lambda: 50,
        sampling_temperature: sampling_temperature,
        num_epochs_per_iteration: 8,
        actor_model+: { hf_model_name: $.episode_generator.initial_model_name_or_path },
        reference_model+: { hf_model_name: $.episode_generator.initial_model_name_or_path},

        general_training_args+: {
            save_steps: 450,
            checkpoint_keep_steps: 450,
            max_seq_len: $.episode_generator.max_sequence_length,
            per_device_train_batch_size: 4,
            gradient_accumulation_steps: null,
        },

        kl_penalty_loss_type: 'control_variate',

        cache_deepspeed_engines: false,  # because it OOM on mila cluster, we can try to cache the engines on microsoft cluster
        move_reference_model_to_cpu: false, # because it OOM on mila cluster, we can try to cache the engines on microsoft cluster

    },

    analyzers: [
    ],
}
 + (import 'sft_deepseekmath_for_MATH_eval.jsonnet')
