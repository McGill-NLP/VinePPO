{
    trainer+: {
        actor_deepspeed_config: (import '../deepspeed/zero_2.jsonnet'),
        critic_deepspeed_config: (import '../deepspeed/zero_2.jsonnet'),
    },
}
