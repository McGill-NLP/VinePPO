local hf_model_name = 'realtreetune/deepseekmath-7b-sft-GSM8K';
local task = (import 'tasks/gsm8k_orig_format.jsonnet');

(import 'polIter_deepseekSft2_dpo_positive_MATH.jsonnet')
+ {
    episode_generator+: {
        // Override the task
        task: task,
        reward_function+: { math_task: $.episode_generator.task },

        initial_model_name_or_path: hf_model_name,

        inference_strategy+: {
            guidance_llm: (import 'guidance_llms/deepseekmath7b-sft-GSM8K.jsonnet') + { api_base: 'none' },
        },
    },
}
+ (import 'sft_deepseekSft_for_gsm8k_eval.jsonnet')
+ (import 'trainers/64_samples.jsonnet')
+ (import 'episode_generators/dpo_positive/maxpair_64.jsonnet')
+ (import 'trainers/dpo_b_0.3.jsonnet')
+ (import 'trainers/3_epochs.jsonnet')