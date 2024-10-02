local tree_expansion_iid = '{{prefix}}{{gen "thinking_process" temperature={temperature} top_p={top_p} top_k={top_k} max_tokens={max_tokens} save_stop_text="stop_text" stop={stop} logprobs={logprobs} n={num_samples}}}
Thus, the answer in few words is: {{gen "chain_of_thought" temperature=0.2 top_p=0.95 top_k=20 max_tokens=20 logprobs={logprobs}}}';

local tree_question_template = '[INST] <<SYS>>
You are a helpful assistant. Always answer in most accurate way.
<</SYS>>
Answer the following factual knowledge based question. Here is an example:
Q: Where in England was Dame Judi Dench born?
A: Dame Judi Dench was born in Heworth, York, England.
Thus, the answer in few words is: Heworth, York, England.

Q: {question}
[/INST]
A: ';

{
    prompt_library: {
        tree: {
            expansion: {
                iid: tree_expansion_iid,
            },
            question_template: tree_question_template,
        },
    },
}
