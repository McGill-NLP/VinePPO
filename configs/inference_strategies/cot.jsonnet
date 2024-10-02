{
    inference_strategy+: {
        type: 'cot',
        samples: 11,
        question_template: $.prompt_library.tree.question_template,
    },
}