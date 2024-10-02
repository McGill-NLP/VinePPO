local tree_expansion_iid = '{{prefix}}{{gen "chain_of_thought" temperature={temperature} max_tokens={max_tokens} save_stop_text="stop_text" stop={stop} logprobs={logprobs} n={num_samples}}}';

local tree_question_template = '[INST] <<SYS>>
You are a helpful assistant. Always answer in most accurate way.
<</SYS>>
Your task is to answer the following question. First write down your thinking process, then choose the correct choice between A, B, C, and D.

Q: {question}
[/INST]';

local tree_answer_extract_next_chat_turn = '{{prefix}}</s><s>[INST]So, based on your thought processes and your choice above, which choice did you choose? Only output A, B, C, D.[/INST] Sure! I chose:
{{gen "final_answer" temperature=0 max_tokens=3}}';
//\n({{#select "final_answer" logprobs="logprobs"}}'+"A{{or}}B{{or}}C{{or}}D"+'{{/select}}';

{
    prompt_library: {
        tree: {
            expansion: {
                iid: tree_expansion_iid,
            },
            question_template: tree_question_template,
            answer_extract: {
                next_chat_turn: tree_answer_extract_next_chat_turn,
            },
        },
    },
}
