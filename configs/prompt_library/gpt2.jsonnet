local tree_expansion_iid = '{{prefix}}{{gen "chain_of_thought" temperature={temperature} max_tokens={max_tokens} save_stop_text="stop_text" seed={seed} n={num_samples}}}';

local tree_question_template = '{query}';

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
