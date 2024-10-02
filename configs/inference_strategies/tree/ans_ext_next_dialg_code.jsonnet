{
    inference_strategy+: {
        answer_extractor+: {
            type: 'next_chat_turn_code',
            program: $.prompt_library.tree.answer_extract.next_chat_turn,
            program_kwargs+: {
            },
        },
    },
}
