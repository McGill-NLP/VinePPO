local orig_library = (import 'llama2.jsonnet');
local tree_expansion_iid = '{{prefix}}{{gen "chain_of_thought" temperature={temperature} top_p={top_p} top_k={top_k} max_tokens={max_tokens} save_stop_text="stop_text" stop={stop} logprobs={logprobs} n={num_samples}}}';

orig_library + {
    prompt_library+: {
        tree+: {
            expansion+: {
                iid: tree_expansion_iid,
            },
        },
    },
}
