local orig_library = (import 'llama2.jsonnet');
local tree_question_template = '{question}';

orig_library + {
    prompt_library+: {
        tree+: {
            question_template: tree_question_template,
        },
    },
}
