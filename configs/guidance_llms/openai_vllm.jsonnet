{
    type: 'openai_vllm',
    caching: false,
    max_calls_per_min: 1e6,
    max_retries: 10,
    api_key: 'EMPTY',
    api_base: std.extVar('APP_OPENAI_VLLM_API_BASE'),
}
