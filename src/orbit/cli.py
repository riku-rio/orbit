from __future__ import annotations

import asyncio

import typer

from orbit import __version__
from orbit.chat import run_chat
from orbit.mcp_client import MCPError, OrbitMCPClient
from orbit.ollama import (
    OllamaClient,
    OllamaError,
    OllamaNotInstalledError,
    OllamaStartError,
    ensure_ollama_running,
)
from orbit.settings import Settings, load_settings, save_settings
from orbit.ui import activity

app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


def select_model(client: OllamaClient) -> str:
    try:
        models = client.list_models()
    except OllamaError as exc:
        typer.echo(f"Error: Could not read Ollama models: {exc}", err=True)
        raise typer.Exit(1) from exc

    if not models:
        typer.echo("No Ollama models found.", err=True)
        typer.echo("Install a model first with: ollama pull <model>", err=True)
        raise typer.Exit(1)

    typer.echo("Select a model:\n")
    for index, model in enumerate(models, start=1):
        typer.echo(f"  {index}. {model}")

    while True:
        choice = typer.prompt(f"\nModel [1-{len(models)}]")
        try:
            index = int(choice)
        except ValueError:
            typer.echo("Enter a model number from the list.", err=True)
            continue

        if 1 <= index <= len(models):
            return models[index - 1]

        typer.echo("Enter a model number from the list.", err=True)


def ensure_model(client: OllamaClient, settings: Settings) -> Settings:
    try:
        models = client.list_models()
    except OllamaError as exc:
        typer.echo(f"Error: Could not read Ollama models: {exc}", err=True)
        raise typer.Exit(1) from exc

    if settings.model in models:
        return settings

    if settings.model is not None:
        typer.echo(f"Configured model '{settings.model}' is not available.\n")

    settings.model = select_model(client)
    save_settings(settings)
    typer.echo(f"\nUsing {settings.model}\n")
    return settings


def _format_setting(value: object | None) -> str:
    return "Default" if value is None else str(value)


def _parse_context_size(value: str) -> int:
    normalized = value.strip().lower()
    multiplier = 1
    if normalized.endswith("k"):
        normalized = normalized[:-1]
        multiplier = 1024
    elif normalized.endswith("m"):
        normalized = normalized[:-1]
        multiplier = 1024 * 1024

    try:
        amount = int(normalized)
    except ValueError as exc:
        raise ValueError("invalid context size") from exc

    context_size = amount * multiplier
    if context_size <= 0:
        raise ValueError("invalid context size")
    return context_size


def edit_context_size(settings: Settings) -> None:
    while True:
        value = typer.prompt(
            "Context size (e.g. 32768 or 32k; press Enter for Default)",
            default="",
            show_default=False,
        ).strip()
        if not value:
            settings.context_size = None
            return
        try:
            context_size = _parse_context_size(value)
        except ValueError:
            typer.echo(
                "Context size must be a positive integer or use k/m (for example 32k).",
                err=True,
            )
            continue
        settings.context_size = context_size
        return


def edit_temperature(settings: Settings) -> None:
    while True:
        value = typer.prompt(
            "Temperature (press Enter for Default)",
            default="",
            show_default=False,
        ).strip()
        if not value:
            settings.temperature = None
            return
        try:
            temperature = float(value)
        except ValueError:
            typer.echo("Temperature must be a non-negative number.", err=True)
            continue
        if temperature < 0:
            typer.echo("Temperature must be a non-negative number.", err=True)
            continue
        settings.temperature = temperature
        return


def open_settings(client: OllamaClient) -> None:
    settings = load_settings()

    while True:
        typer.echo("\norbit Settings\n")
        typer.echo(f"  1. Model          {settings.model or 'Not set'}")
        typer.echo(f"  2. Context size   {_format_setting(settings.context_size)}")
        typer.echo(f"  3. Temperature    {_format_setting(settings.temperature)}")
        typer.echo("\n  0. Exit")

        choice = typer.prompt("\nSelect setting").strip()
        if choice == "0":
            return
        if choice == "1":
            prepare_ollama(client)
            settings.model = select_model(client)
        elif choice == "2":
            edit_context_size(settings)
        elif choice == "3":
            edit_temperature(settings)
        else:
            typer.echo("Select a setting from the list.", err=True)
            continue

        save_settings(settings)
        typer.echo("Settings saved.")


def prepare_ollama(client: OllamaClient) -> None:
    if client.is_running():
        return

    try:
        with activity("Starting Ollama..."):
            ensure_ollama_running(client)
    except OllamaNotInstalledError:
        typer.echo("Error: Ollama is not installed.", err=True)
        typer.echo("Install Ollama and try again.", err=True)
        raise typer.Exit(1)
    except OllamaStartError as exc:
        typer.echo("Error: Failed to start Ollama.", err=True)
        typer.echo(str(exc), err=True)
        typer.echo("Start Ollama manually and try again.", err=True)
        raise typer.Exit(1) from exc


async def _run_chat_with_mcp(
    client: OllamaClient,
    settings: Settings,
    *,
    render_markdown: bool | None,
) -> None:
    try:
        async with OrbitMCPClient() as mcp_client:
            await run_chat(
                client,
                settings,
                mcp_client,
                render_markdown=render_markdown,
            )
    except MCPError as exc:
        typer.echo(f"Error: {exc}", err=True)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show the orbit version and exit.",
    ),
    settings: bool = typer.Option(
        False,
        "--settings",
        help="Open orbit global settings.",
    ),
    plain: bool = typer.Option(
        False,
        "--plain",
        help="Stream assistant responses as raw Markdown.",
    ),
) -> None:
    client = OllamaClient()

    if settings:
        open_settings(client)
        raise typer.Exit()

    prepare_ollama(client)
    current_settings = ensure_model(client, load_settings())
    asyncio.run(
        _run_chat_with_mcp(
            client,
            current_settings,
            render_markdown=False if plain else None,
        )
    )
