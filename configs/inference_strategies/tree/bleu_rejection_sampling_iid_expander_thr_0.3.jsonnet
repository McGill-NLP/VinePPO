(import 'bleu_rejection_sampling_iid_expander.jsonnet')
+ {
    inference_strategy+: {
            node_expander+: {
              bleu_acc_threshold: 0.3,
            },
        },
}