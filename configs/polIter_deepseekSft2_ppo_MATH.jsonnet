local hf_model_name = 'realtreetune/deepseekmath-7b-sft-MATH-v2';

local actor_tokenizer = {
    type: 'pretrained',
    hf_model_name: hf_model_name,
};

local math_task = (import 'tasks/math_inplace_no_answer_prefix.jsonnet') + {
    prepend_in_context_few_shot: false,
    ensure_fit_in_context_size: false,
};

local num_episodes_per_iteration = 512;
local num_rollouts_per_sample = 8;
local num_dataset_samples_per_iteration = num_episodes_per_iteration / num_rollouts_per_sample;
local total_num_iterations = 1000;

local sampling_temperature = 0.6;

(import 'gvar.jsonnet')
+ (import 'prompt_library/MATH_step_by_step_sft.jsonnet')
+ (import 'runtimes/policy_iteration.jsonnet')
+ (import 'episode_generators/math_episode_generator.jsonnet')
+ (import 'trainers/ppo_MATH.jsonnet')
+ {
    episode_generator+: {
        // Override the task
        task: math_task,
        reward_function+: { math_task: $.episode_generator.task },
        reasoning_step_delimiter: '',
        answer_prefix: null,

        initial_model_name_or_path: hf_model_name,

        dataset_sample_with_replacement: true,
        dataset_num_samples_per_iteration: num_dataset_samples_per_iteration,
        total_num_iterations: $.num_iterations,

        max_sequence_length: 2499,  // Increase the max_seq_len since the model context size is 4096

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

            no_cache: true,
        },
    },

    tokenizer: actor_tokenizer,
    use_deepspeed: true,

    num_iterations: total_num_iterations,
    num_episodes_per_iteration: num_episodes_per_iteration,
    episodes_cloud_log_steps: 50,

    trainer+: {
        params+: { temperature: $.episode_generator.inference_strategy.node_expander.program_kwargs.temperature },

        actor_model+: { hf_model_name: $.episode_generator.initial_model_name_or_path },
        critic_model+: { pretrained_backbone_model+: { hf_model_name: $.episode_generator.initial_model_name_or_path } },
        reference_model+: { hf_model_name: $.episode_generator.initial_model_name_or_path },

        // To prevent OOM errors
        report_entropy: false,

        general_training_args+: {
            save_steps: 30,
            checkpoint_keep_steps: 60,
        },
    },

    analyzers: [
        (import 'analyzers/valnet_prediction.jsonnet') + {
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

        (import 'analyzers/valnet_action_ranking.jsonnet') + {
            task: $.episode_generator.task,
            tokenizer: $.tokenizer,
            vllm_server+: { swap_space: 64 },

            reward_function: $.episode_generator.reward_function,

            max_num_requests: 512,
            max_num_states: 128,

            append_bos_to_query: $.episode_generator.append_bos_to_query,

            inference_strategy+: {
                guidance_llm: $.episode_generator.inference_strategy.guidance_llm,

                // Small model. Can afford more concurrent programs.
                max_concurrent_programs: 32,
                max_concurrent_generations: 16,

                node_expander+: {
                    program_kwargs+: { temperature: $.episode_generator.inference_strategy.node_expander.program_kwargs.temperature },
                    model_context_size: $.episode_generator.inference_strategy.node_expander.model_context_size,
                    tokenizer: $.tokenizer,
                },
            },
        },
    ],
}
+ (import 'sft_deepseekmath_for_MATH_eval.jsonnet')
+ (import 'trainers/lam1.jsonnet')
+ (import 'trainers/refKl0.0001.jsonnet')
+ (import 'trainers/klLoss.jsonnet')
