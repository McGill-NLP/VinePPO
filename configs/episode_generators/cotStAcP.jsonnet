(import 'treeStAcP.jsonnet') + {
    episode_generator+: {
        branch_factor: 1,
    },

    trainer+: {
        training_args+: {
            // This learning has been selected from hyperparameter search
            // on GSM8K validation set.Search space: 3e-5, 1e-5, 3e-6, 1e-6, 3e-7
            learning_rate: 3e-6,


            save_steps: 750,
            checkpoint_keep_steps: 1500,
            gradient_accumulation_steps: 16,
        },
    },
}
