{
    trainer+: {
        type: "mle",
        num_epochs_per_iteration: 2,
        training_args: (import "training_args.jsonnet"),
    }
}