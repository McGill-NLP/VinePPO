local orig_library = (import 'llama2.jsonnet');
local tree_expansion_iid = '{{prefix}}{{gen "chain_of_thought" temperature={temperature} max_tokens={max_tokens} save_stop_text="stop_text" stop={stop} logprobs={logprobs} n={num_samples}}}';

local tree_answer_extract_next_chat_turn = '{{prefix}}</s><s>[INST]So, what is the final answer? Only output digits.[/INST] Sure! The final answer is:
{{gen "final_answer" temperature={temperature} max_tokens={max_tokens}}}';

local tree_question_template = '[INST] <<SYS>>
You are a helpful assistant solving math questions. Always answer in most accurate way.
<</SYS>>
Answer the following middle school math word problems, which require multi-step arithmetic reasoning.

Q: {question}[/INST]
A:';

orig_library + {
    prompt_library+: {
        tree+: {
            expansion+: {
                iid: tree_expansion_iid,
            },
            answer_extract+: {
                next_chat_turn: tree_answer_extract_next_chat_turn,
            },
            question_template: tree_question_template,
        },
    },
}
