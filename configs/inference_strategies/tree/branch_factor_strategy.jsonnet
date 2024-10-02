{
    inference_strategy+: {
        node_expander+: {
            branch_factor_strategy: {
                type: 'list',
                branch_factors: [
                    { depth: 0, branch_factor: 3 },
                    { depth: 6, branch_factor: 1 },
                ],
            },
        },
    },
}
