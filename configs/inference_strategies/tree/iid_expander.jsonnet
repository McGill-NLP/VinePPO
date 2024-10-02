{
    inference_strategy+: {
        node_expander+: {
            type: 'iid',
            program: $.prompt_library.tree.expansion.iid,
            program_kwargs: {
                temperature: 0.95,
                top_p: 0.9,
                top_k: 50,
                max_tokens: 256,
                stop_regex: '"\nStep"',
            },
            node_text_template: 'Step {chain_of_thought}',
        },
    },
}
