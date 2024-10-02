(import 'iid_expander.jsonnet')
+
{
    inference_strategy+: {
        node_expander+: {
            program_kwargs+: {
                temperature: 0.3,
            },
        },
    },
}
