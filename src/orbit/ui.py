from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum

from rich.align import Align
from rich.cells import cell_len
from rich.console import Console, Group, RenderableType
from rich.control import Control
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.segment import ControlType
from rich.text import Text

console = Console()

CONCISE_PROMPT_LIMIT = 60
THINKING_PREVIEW_LIMIT = 240
_ERASE_LINE = Control((ControlType.ERASE_IN_LINE, 2))


class ViewMode(str, Enum):
    CONCISE = "concise"
    FULL = "full"

    def toggled(self) -> "ViewMode":
        return ViewMode.FULL if self is ViewMode.CONCISE else ViewMode.CONCISE


@contextmanager
def activity(description: str) -> Iterator[None]:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
    )
    with progress:
        progress.add_task(description, total=None)
        yield


def _display_prompt(prompt: str, mode: ViewMode) -> str:
    if mode is ViewMode.FULL or len(prompt) <= CONCISE_PROMPT_LIMIT:
        return prompt
    return f"{prompt[:CONCISE_PROMPT_LIMIT].rstrip()}..."


def request_header(prompt: str, mode: ViewMode) -> Text:
    request = Text("> ", style="bold")
    request.append(_display_prompt(prompt, mode))
    return request


def duration_footer(duration_seconds: float) -> Text:
    footer = Text("↳ ", style="dim")
    footer.append(f"{duration_seconds:.2f}s", style="dim")
    return footer


def session_status(
    model: str,
    context_used: int,
    context_total: int | None,
) -> Align:
    status = Text(style="dim")
    status.append(model, style="bold dim")
    status.append("  •  ")

    if context_total is None:
        status.append("Context unavailable")
    else:
        used = max(0, min(context_used, context_total))
        remaining = max(context_total - used, 0)
        status.append(f"Context {used:,} / {context_total:,}")
        status.append("  •  ")
        status.append(f"{remaining:,} left")

    return Align.center(status)


def clear_typed_prompt(prompt: str) -> None:
    if not console.is_terminal:
        return

    width = max(console.width, 1)
    cells = max(cell_len(f"> {prompt}"), 1)
    rows = max(1, (cells + width - 1) // width)

    console.control(Control.move(0, -rows), Control.move_to_column(0))
    for index in range(rows):
        console.control(_ERASE_LINE)
        if index < rows - 1:
            console.control(Control.move(0, 1), Control.move_to_column(0))
    if rows > 1:
        console.control(Control.move(0, -(rows - 1)), Control.move_to_column(0))


def clear_rendered_block(rows: int) -> None:
    if not console.is_terminal or rows <= 0:
        return

    console.control(Control.move(0, -rows), Control.move_to_column(0))
    for index in range(rows):
        console.control(_ERASE_LINE)
        if index < rows - 1:
            console.control(Control.move(0, 1), Control.move_to_column(0))
    if rows > 1:
        console.control(Control.move(0, -(rows - 1)), Control.move_to_column(0))


def render_height(renderable: RenderableType) -> int:
    return max(1, len(console.render_lines(renderable, console.options, pad=False)))


def generation_progress() -> Progress:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        auto_refresh=False,
    )
    progress.add_task("Generating response...", total=None)
    return progress


def generation_renderable(
    progress: Progress,
    thinking: str,
    mode: ViewMode,
) -> RenderableType:
    del mode
    items: list[RenderableType] = [progress.get_renderable()]
    if thinking:
        shown = " ".join(thinking.split())
        if len(shown) > THINKING_PREVIEW_LIMIT:
            shown = f"...{shown[-THINKING_PREVIEW_LIMIT:]}"
        label = Text("Thinking: ", style="bold dim", no_wrap=True, overflow="ellipsis")
        label.append(shown, style="dim")
        items.append(label)
    return Group(*items)
