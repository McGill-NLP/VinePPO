local math_task = (import '../tasks/math.jsonnet') + {
    prepend_in_context_few_shot: false,
    ensure_fit_in_context_size: false,
};

local prompt_library = (import '../prompt_library/MATH_step_by_step_sft.jsonnet');
local question_template = prompt_library.prompt_library.tree.question_template;

{
    episode_generator+: {
        type: 'math_episode_generator',
        vllm_server+: {
            swap_space: 16,
        },

        append_bos_to_query: true,
        append_eos_to_response: true,

        dataset_shuffle_on_each_iteration: true,
        dataset_shuffle_before_portion: true,
        dataset_sample_with_replacement: false,

        vllm_gpu_memory_utilization: 'auto',
        vllm_min_available_gpu_memory_mb: 20 * 1024,
        wait_until_memory_release: true,

        reward_function: {
            type: 'math_reward_function',
            penalize_unfinished_response: true,
            unfinished_response_penalty: 0.0,
            math_task: $.episode_generator.task,
        },
        reasoning_step_delimiter: '\n',
        answer_prefix: '\n\n# Answer\n',

        max_sequence_length: 2048,
        max_question_length: 1512,
        question_template: question_template,

        fill_missing_episodes: true,

        task: math_task,
    },
}
