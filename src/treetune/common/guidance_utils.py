import guidance


async def run_program(guidance_template, **kwargs):
    program = guidance(
        guidance_template,
        silent=True,
        logging=True,
        async_mode=True,
        stream=False,
        await_missing=True,
    )
    result = await program(**kwargs)
    return result
