{
    trainer+: {
        type: "value_network",
        num_epochs_per_iteration: 5,
        training_args: (import "training_args.jsonnet"),
    }
}