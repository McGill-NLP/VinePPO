(import 'iid_expander.jsonnet')
+ {
    inference_strategy+: {
            node_expander+: {
              type: 'bleu_rejection_sampling_iid',
              max_try: 20,
              bleu_acc_threshold: 0.5,
            },
        },
}