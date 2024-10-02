local hf_model_name = 'realtreetune/rho-1b-sft-MATH';

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

        vllm_server+: { swap_space: 8, max_num_seqs: 512 },
        vllm_min_available_gpu_memory_mb: 10 * 1024,

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
                model_context_size: 2047,
                tokenizer: $.tokenizer,
            },

            answer_extractor: {
                type: 'identity',
                node_key_name: 'text',
            },

            guidance_llm: (import 'guidance_llms/rho1b-sft-MATH.jsonnet') + { api_base: 'none' },

            question_field: 'query',
            question_template: $.prompt_library.tree.question_template,

            no_cache: true,
        },
    },

    tokenizer: {
        type: 'pretrained',
        hf_model_name: $.episode_generator.initial_model_name_or_path,
    },
    use_deepspeed: true,

    num_iterations: total_num_iterations,
    num_episodes_per_iteration: num_episodes_per_iteration,
    episodes_cloud_log_steps: 50,

    trainer+: {
        params+: { temperature: $.episode_generator.inference_strategy.node_expander.program_kwargs.temperature },

        actor_model+: { hf_model_name: $.episode_generator.initial_model_name_or_path },
        critic_model+: { pretrained_backbone_model+: { hf_model_name: $.episode_generator.initial_model_name_or_path } },
        reference_model+: { hf_model_name: $.episode_generator.initial_model_name_or_path },

        actor_deepspeed_config: (import 'deepspeed/zero_0.jsonnet'),
        critic_deepspeed_config: (import 'deepspeed/zero_0.jsonnet'),

        // To prevent OOM errors
        report_entropy: false,

        general_training_args+: {
            target_train_batch_size: 64,
            per_device_train_batch_size: null,  // Will be auto computed
            gradient_accumulation_steps: 1,

            save_steps: 40,
            checkpoint_keep_steps: 40,
        },
    },


    analyzers: [
        (import 'analyzers/valnet_prediction.jsonnet') + {
            task: $.episode_generator.task,
            tokenizer: $.tokenizer,
            vllm_server+: { swap_space: 24 },

            reward_function: $.episode_generator.reward_function,

            // Small model. Can afford more requests.
            max_num_requests: 1024,

            inference_strategy+: {
                guidance_llm: $.episode_generator.inference_strategy.guidance_llm,

                // Small model. Can afford more concurrent programs.
                max_concurrent_programs: 128,
                max_concurrent_generations: 128,

                node_expander+: {
                    program_kwargs+: { temperature: $.episode_generator.inference_strategy.node_expander.program_kwargs.temperature },
                    model_context_size: $.episode_generator.inference_strategy.node_expander.model_context_size,
                    tokenizer: $.tokenizer,
                },
            },
        },

        (import 'analyzers/ppo_grad_variance.jsonnet') + {
            per_device_batch_size: 16,
        },

        (import 'analyzers/valnet_action_ranking.jsonnet') + {
            task: $.episode_generator.task,
            tokenizer: $.tokenizer,
            vllm_server+: { swap_space: 24 },

            reward_function: $.episode_generator.reward_function,

            max_num_requests: 512,
            max_num_states: 256,

            append_bos_to_query: $.episode_generator.append_bos_to_query,

            inference_strategy+: {
                guidance_llm: $.episode_generator.inference_strategy.guidance_llm,

                // Small model. Can afford more concurrent programs.
                max_concurrent_programs: 128,
                max_concurrent_generations: 128,

                node_expander+: {
                    program_kwargs+: { temperature: $.episode_generator.inference_strategy.node_expander.program_kwargs.temperature },
                    model_context_size: $.episode_generator.inference_strategy.node_expander.model_context_size,
                    tokenizer: $.tokenizer,
                },
            },
        },
    ],
}
+ (import 'sft_rho1b_for_MATH_eval.jsonnet')
+ (import 'trainers/lam1.jsonnet')
+ (import 'trainers/refKl0.0001.jsonnet')
+ (import 'trainers/klLoss.jsonnet')
