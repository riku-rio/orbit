from __future__ import annotations

import re
from dataclasses import dataclass, field

from rich.console import Console
from rich.markdown import Markdown

_FENCE_START = re.compile(r"^ {0,3}(?P<marker>`{3,}|~{3,})(?P<info>.*)$")


@dataclass
class MarkdownBuffer:
    """Collect streamed text until a Markdown block is safe to render."""

    _pending_text: str = ""
    _block_lines: list[str] = field(default_factory=list)
    _fence_char: str | None = None
    _fence_length: int = 0

    def feed(self, text: str) -> list[str]:
        if not text:
            return []

        self._pending_text += text
        blocks: list[str] = []

        while True:
            newline = self._pending_text.find("\n")
            if newline < 0:
                break

            line = self._pending_text[: newline + 1]
            self._pending_text = self._pending_text[newline + 1 :]
            blocks.extend(self._consume_line(line))

        return blocks

    def finish(self) -> list[str]:
        blocks: list[str] = []
        if self._pending_text:
            blocks.extend(self._consume_line(self._pending_text))
            self._pending_text = ""

        block = self._flush_block()
        if block is not None:
            blocks.append(block)
        return blocks

    def _consume_line(self, line: str) -> list[str]:
        line_body = line.rstrip("\r\n")

        if self._fence_char is not None:
            self._block_lines.append(line)
            if self._is_closing_fence(line_body):
                self._fence_char = None
                self._fence_length = 0
                block = self._flush_block()
                return [block] if block is not None else []
            return []

        self._block_lines.append(line)
        opening = _FENCE_START.match(line_body)
        if opening is not None:
            marker = opening.group("marker")
            self._fence_char = marker[0]
            self._fence_length = len(marker)
            return []

        if not line_body.strip():
            block = self._flush_block()
            return [block] if block is not None else []

        return []

    def _is_closing_fence(self, line: str) -> bool:
        if self._fence_char is None:
            return False

        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if indent > 3:
            return False

        marker_length = 0
        for char in stripped:
            if char != self._fence_char:
                break
            marker_length += 1

        if marker_length < self._fence_length:
            return False
        return not stripped[marker_length:].strip()

    def _flush_block(self) -> str | None:
        if not self._block_lines:
            return None

        block = "".join(self._block_lines)
        self._block_lines.clear()
        if not block.strip():
            return None
        return block


class MarkdownStreamRenderer:
    """Render complete Markdown blocks while preserving raw streamed content."""

    def __init__(
        self,
        console: Console,
        render_markdown: bool | None = None,
    ) -> None:
        self.console = console
        self.render_markdown = console.is_terminal if render_markdown is None else render_markdown
        self.buffer = MarkdownBuffer()
        self.started = False
        self.finished = False
        self.wrote_output = False
        self._output_ends_with_newline = True

    def start(self) -> None:
        if self.finished:
            raise RuntimeError("Cannot restart a finished renderer.")
        self.started = True

    def feed(self, text: str) -> None:
        if self.finished:
            raise RuntimeError("Cannot feed a finished renderer.")
        if not text:
            return
        if not self.started:
            self.start()

        if not self.render_markdown:
            self._write_raw(text)
            return

        for block in self.buffer.feed(text):
            self._render_block(block)

    def finish(self) -> None:
        if self.finished:
            return

        if self.render_markdown:
            for block in self.buffer.finish():
                self._render_block(block)

        self.finished = True

    def abort(self) -> None:
        self.finish()

    def ensure_line_break(self) -> None:
        if not self.wrote_output or self._output_ends_with_newline:
            return
        self.console.file.write("\n")
        self.console.file.flush()
        self._output_ends_with_newline = True

    def _write_raw(self, text: str) -> None:
        self.console.file.write(text)
        self.console.file.flush()
        self.wrote_output = True
        self._output_ends_with_newline = text.endswith(("\n", "\r"))

    def _render_block(self, block: str) -> None:
        self.console.print(Markdown(block))
        self.wrote_output = True
        self._output_ends_with_newline = True
