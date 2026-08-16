from __future__ import annotations

from time import perf_counter
from typing import Any

import typer
from rich.live import Live

from orbit.mcp_client import MCPError, OrbitMCPClient
from orbit.memory.policy import MEMORY_SYSTEM_PROMPT
from orbit.ollama import ChatChunk, OllamaClient, OllamaError, ToolCall
from orbit.rendering import MarkdownStreamRenderer
from orbit.settings import Settings
from orbit.ui import (
    ViewMode,
    activity,
    clear_rendered_block,
    clear_typed_prompt,
    console,
    duration_footer,
    generation_progress,
    generation_renderable,
    render_height,
    request_header,
    session_status,
    tool_completed,
    tool_failed,
    tool_started,
)

EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit"}
VIEW_COMMAND = "/view"
HELP_COMMAND = "/help"
MAX_TOOL_ROUNDS = 16


def ensure_model_loaded(client: OllamaClient, settings: Settings) -> int | None:
    if settings.model is None:
        raise ValueError("A model must be selected before loading it.")

    loaded_context = client.model_context_length(settings.model)
    if loaded_context is not None and (
        settings.context_size is None or loaded_context == settings.context_size
    ):
        return loaded_context

    with activity(f"Loading {settings.model}..."):
        client.load_model(settings.model, settings.ollama_load_options())

    return client.model_context_length(settings.model)


def _context_used(chunk: ChatChunk | None, context_total: int | None) -> int | None:
    if chunk is None or chunk.prompt_eval_count is None:
        return None

    used = chunk.prompt_eval_count + (chunk.eval_count or 0)
    if context_total is not None:
        return min(used, context_total)
    return used


def _show_help() -> None:
    typer.echo("\nCommands:\n")
    typer.echo("  /help        Show available commands")
    typer.echo("  /view        Toggle concise/full view")
    typer.echo("  /exit        Exit orbit")
    typer.echo("  /quit        Exit orbit\n")


def _show_status(model: str, context_used: int, context_total: int | None) -> None:
    console.print(session_status(model, context_used, context_total))


def _show_tool_error(message: str) -> None:
    detail = " ".join(message.split())
    if detail:
        console.print(f"  {detail}", style="red")


def _assistant_message(
    thinking: str,
    content: str,
    tool_calls: list[ToolCall],
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }
    if thinking:
        message["thinking"] = thinking
    if tool_calls:
        message["tool_calls"] = [call.as_ollama_message() for call in tool_calls]
    return message


def _tool_message(
    tool_name: str,
    content: str,
    images: tuple[str, ...] = (),
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "tool",
        "tool_name": tool_name,
        "content": content,
    }
    if images:
        message["images"] = list(images)
    return message


async def run_chat(
    client: OllamaClient,
    settings: Settings,
    mcp_client: OrbitMCPClient,
    *,
    render_markdown: bool | None = None,
) -> None:
    if settings.model is None:
        raise ValueError("A model must be selected before starting chat.")

    try:
        tools = await mcp_client.list_ollama_tools()
    except MCPError as exc:
        typer.echo(f"Error: {exc}", err=True)
        return

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": MEMORY_SYSTEM_PROMPT}
    ]
    view_mode = ViewMode.CONCISE
    context_used = 0

    try:
        context_total = ensure_model_loaded(client, settings)
    except OllamaError as exc:
        typer.echo(f"Error: Could not load {settings.model}: {exc}", err=True)
        return

    typer.echo("Type /help for commands.\n")
    _show_status(settings.model, context_used, context_total)

    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            return

        if not prompt:
            continue
        lowered = prompt.lower()
        if lowered in EXIT_COMMANDS:
            return
        if lowered == HELP_COMMAND:
            _show_help()
            _show_status(settings.model, context_used, context_total)
            continue
        if lowered == VIEW_COMMAND:
            view_mode = view_mode.toggled()
            typer.echo(f"View mode: {view_mode.value}\n")
            _show_status(settings.model, context_used, context_total)
            continue
        if prompt.startswith("/"):
            command = prompt.split(maxsplit=1)[0]
            typer.echo(f"Unknown command: {command}", err=True)
            typer.echo("Type /help for commands.\n", err=True)
            _show_status(settings.model, context_used, context_total)
            continue

        clear_typed_prompt(prompt)
        console.print(request_header(prompt, view_mode))

        turn_message_start = len(messages)
        messages.append({"role": "user", "content": prompt})

        final_chunk: ChatChunk | None = None
        renderer = MarkdownStreamRenderer(console, render_markdown=render_markdown)
        tool_rounds = 0
        generation_duration_ns = 0
        turn_started_at = perf_counter()

        try:
            loaded_context = ensure_model_loaded(client, settings)
            if loaded_context is not None:
                context_total = loaded_context

            while True:
                thinking_parts: list[str] = []
                content_parts: list[str] = []
                tool_calls: list[ToolCall] = []
                round_final_chunk: ChatChunk | None = None
                answer_started = False

                live: Live | None = None
                live_active = False
                live_rows = 0

                if console.is_terminal:
                    progress = generation_progress()
                    renderable = generation_renderable(progress, "", view_mode)
                    live_rows = render_height(renderable)
                    live = Live(
                        renderable,
                        console=console,
                        refresh_per_second=12,
                        transient=False,
                        vertical_overflow="ellipsis",
                    )
                    live.start()
                    live_active = True

                try:
                    for chunk in client.chat(
                        settings.model,
                        messages,
                        settings.ollama_options(),
                        tools=tools,
                    ):
                        round_final_chunk = chunk

                        if chunk.thinking and not answer_started:
                            thinking_parts.append(chunk.thinking)
                            if live_active and live is not None:
                                renderable = generation_renderable(
                                    progress,
                                    "".join(thinking_parts),
                                    view_mode,
                                )
                                live_rows = render_height(renderable)
                                live.update(renderable, refresh=True)

                        if chunk.tool_calls:
                            tool_calls.extend(chunk.tool_calls)

                        if chunk.content:
                            if not answer_started:
                                answer_started = True
                                if live_active and live is not None:
                                    live.stop()
                                    live_active = False
                                    clear_rendered_block(live_rows)
                                renderer.start()
                            content_parts.append(chunk.content)
                            renderer.feed(chunk.content)
                finally:
                    if live_active and live is not None:
                        live.stop()
                        clear_rendered_block(live_rows)

                if (
                    round_final_chunk is not None
                    and round_final_chunk.total_duration_ns is not None
                ):
                    generation_duration_ns += round_final_chunk.total_duration_ns
                final_chunk = round_final_chunk

                thinking = "".join(thinking_parts)
                content = "".join(content_parts)
                messages.append(_assistant_message(thinking, content, tool_calls))

                if not tool_calls:
                    break

                tool_rounds += 1
                if tool_rounds > MAX_TOOL_ROUNDS:
                    raise OllamaError(
                        f"Tool call limit exceeded ({MAX_TOOL_ROUNDS} rounds)."
                    )

                if renderer.wrote_output:
                    renderer.ensure_line_break()

                for call in tool_calls:
                    console.print(tool_started(call.name))
                    started_at = perf_counter()
                    result_images: tuple[str, ...] = ()
                    try:
                        result = await mcp_client.call_tool(call.name, call.arguments)
                    except MCPError as exc:
                        elapsed = perf_counter() - started_at
                        console.print(tool_failed(elapsed))
                        result_content = f"Tool error: {exc}"
                        _show_tool_error(result_content)
                    else:
                        elapsed = perf_counter() - started_at
                        if result.is_error:
                            console.print(tool_failed(elapsed))
                            _show_tool_error(result.content)
                        else:
                            console.print(tool_completed(elapsed))
                        result_content = result.content
                        result_images = result.images

                    messages.append(
                        _tool_message(
                            call.name,
                            result_content,
                            result_images,
                        )
                    )

                typer.echo()
        except KeyboardInterrupt:
            del messages[turn_message_start:]
            renderer.abort()
            if renderer.wrote_output:
                renderer.ensure_line_break()
            else:
                typer.echo()
            return
        except OllamaError as exc:
            del messages[turn_message_start:]
            renderer.abort()
            if renderer.wrote_output:
                renderer.ensure_line_break()
            typer.echo(f"Error: {exc}", err=True)
            typer.echo()
            _show_status(settings.model, context_used, context_total)
            continue

        renderer.finish()
        renderer.ensure_line_break()

        measured_context = _context_used(final_chunk, context_total)
        if measured_context is not None:
            context_used = measured_context

        duration_seconds = (
            generation_duration_ns / 1_000_000_000
            if generation_duration_ns > 0
            else perf_counter() - turn_started_at
        )
        console.print(duration_footer(duration_seconds))
        typer.echo()
        _show_status(settings.model, context_used, context_total)
