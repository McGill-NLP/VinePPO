{
    inference_strategy+: {
        type: 'tree',
        max_depth: 1,
        question_template: $.prompt_library.tree.question_template,
    },
} + (import "tree/branch_factor_strategy.jsonnet")