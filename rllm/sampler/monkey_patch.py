import asyncio
from typing import Any

import uvicorn
from fastapi import FastAPI

from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.utils import find_process_using_port,set_ulimit
from vllm.version import __version__ as VLLM_VERSION
from vllm.entrypoints.openai.api_server import (
    TIMEOUT_KEEP_ALIVE,
    create_server_socket,
    build_async_engine_client,
    init_app_state,
    serve_http,
    build_app,
)
from vllm.entrypoints.launcher import _add_shutdown_handlers

async def serve_http(app: FastAPI, **uvicorn_kwargs: Any):
    print("Available routes are:")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

        print("Route: %s, Methods: %s", path, ', '.join(methods))

    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)
    _add_shutdown_handlers(app, server)

    loop = asyncio.get_running_loop()

    server_task = loop.create_task(server.serve())

    async def dummy_shutdown() -> None:
        pass

    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        port = uvicorn_kwargs["port"]
        process = find_process_using_port(port)
        if process is not None:
            print(
                "port %s is used by process %s launched with command:\n%s",
                port, process, " ".join(process.cmdline()))
        print("Shutting down FastAPI HTTP server.")
        return server.shutdown()



async def run_server(args, **uvicorn_kwargs) -> None:
    print("vLLM API server version %s", VLLM_VERSION)
    print("args: %s", args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    valide_tool_parses = ToolParserManager.tool_parsers.keys()
    if args.enable_auto_tool_choice \
        and args.tool_call_parser not in valide_tool_parses:
        raise KeyError(f"invalid tool call parser: {args.tool_call_parser} "
                       f"(chose from {{ {','.join(valide_tool_parses)} }})")

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()

    async with build_async_engine_client(args) as engine_client:
        app = build_app(args)

        model_config = await engine_client.get_model_config()
        init_app_state(engine_client, model_config, app.state, args)

        shutdown_task = await serve_http(
            app,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    await shutdown_task

    sock.close()