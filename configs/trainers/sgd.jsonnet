{
    trainer+: {
        training_args+: {
            optim: 'sgd',
            learning_rate: 3e-6,
            sgd_momentum: 0.9,
        },
    },
}


{
    trainer+: {
        deepspeed_config+:{
            optimizer: (import '../deepspeed/sgd_optimizer.jsonnet'),
        }
    },
}


