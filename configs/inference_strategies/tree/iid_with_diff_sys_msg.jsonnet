{
    inference_strategy+: {
        node_expander+: {
            type: 'iid_with_different_system_message',
            system_messages: [
                'You are a helpful assistant solving math questions. Always answer in most accurate way.',
                'You are a creative assistant solving math questions. Try to always answer with clever novel ways.',
                'You are a pessimistic assistant solving math questions. Make sure the answers are correct.',
                'You are a high-school teacher showing how to solve math questions. Make sure the answers are correct and informative.',
                'You are a undergrad professor showing how to solve math questions. Make sure the answers are correct and informative.',
            ],
            sys_msg_regex: @'\[INST\] <<SYS>>\s*(.*?)\s*<<\/SYS>>',
        },
    },
}
