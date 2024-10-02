{
    inference_strategy+: {
        answer_extractor+: {
            type: 'next_chat_turn',
            program: $.prompt_library.tree.answer_extract.next_chat_turn,
            program_kwargs: {
                temperature: 0,
                max_tokens: 20,
            },
        },
    },
}
