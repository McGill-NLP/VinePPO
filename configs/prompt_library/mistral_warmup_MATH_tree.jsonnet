local tree_expansion_iid = '{{prefix}}\n{{gen "chain_of_thought" temperature={temperature} top_p={top_p} top_k={top_k} max_tokens={max_tokens} save_stop_text="stop_text" stop_regex={stop_regex}}}';
local tree_expansion_iid_after_branching = '{{prefix}}\n{{gen "chain_of_thought" temperature={temperature} top_p={top_p} top_k={top_k} max_tokens={max_tokens}}}';
local tree_question_template = '[MATH_TASK] Problem:
{query}

Solution:';

{
    prompt_library+: {
        tree+: {
            expansion+: {
                iid: tree_expansion_iid,
                iid_after_branching: tree_expansion_iid_after_branching,
            },
            question_template: tree_question_template,
        },
    },
}
