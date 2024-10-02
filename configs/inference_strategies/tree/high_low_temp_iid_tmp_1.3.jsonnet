(import 'high_low_temp_iid.jsonnet')
+
{
    inference_strategy+: {
        node_expander+: {
            program_kwargs+: {
                cot1_temperature: 1.3,
            },
        },
    },
}
