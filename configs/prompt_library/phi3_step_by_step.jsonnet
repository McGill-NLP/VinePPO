local tree_expansion_iid = '{{prefix}}{{gen "chain_of_thought" temperature={temperature} top_p={top_p} max_tokens={max_tokens} save_stop_text="stop_text" stop={stop} n={num_samples}}}';
local tree_question_template = '<|system|> You are a helpful AI assistant.<|end|><|user|> {query}
Please reason step by step, and put your final answer within \\boxed{{}}.<|end|><|assistant|>';

{
    prompt_library+: {
        tree+: {
            expansion+: {
                iid: tree_expansion_iid,
            },
            question_template: tree_question_template,
        },
    },
}
