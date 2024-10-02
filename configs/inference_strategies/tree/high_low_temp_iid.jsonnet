{
    inference_strategy+: {
        node_expander+: {
            type: 'high_low_temperature_iid',
            program: $.prompt_library.tree.expansion.high_low_temperature_iid,
            program_kwargs: {
                cot1_temperature: 1.3,
                cot1_top_p: 0.9,
                cot1_max_tokens: 10,
                cot2_temperature: 0.9,
                cot2_top_p: 0.95,
                cot2_top_k: 50,
                cot2_max_tokens: 256,
                cot2_stop_regex: '"\nStep"',
            },
            node_text_template: 'Step {chain_of_thought_1}{chain_of_thought_2}',
        },
    },
}
