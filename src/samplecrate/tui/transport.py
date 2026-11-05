"""The transport bar: real, clickable playback controls, shared by the main screen, the Finder and the bank builder."""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static


class Control(Static):
    """A label you can click to run an app action. It takes no focus, so it doesn't interrupt the keyboard."""

    def __init__(self, label: str, action: str, **kwargs):
        super().__init__(label, **kwargs)
        self.action_name = action
        self._label = label

    async def on_click(self, event) -> None:
        event.stop()
        await self.app.run_action(self.action_name)

    def show(self, label: str, *, lit: bool = False, enabled: bool = True) -> None:
        if label != self._label:
            self._label = label
            self.update(label)
        self.set_class(lit, "-lit")
        self.set_class(not enabled, "-idle")


@dataclass(frozen=True)
class TransportState:
    playing: bool
    position: str
    duration: str
    play_on_select: bool
    loop: bool
    volume: int
    view: str


class TransportBar(Horizontal):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Held directly, because the frame clock can update the bar before it's mounted or while the app is shutting down
        self.play = Control("▶ Play", "toggle_play", classes="control play")
        self.stop = Control("■ Stop", "stop_audio", classes="control stop")
        self.time = Static("", classes="time")
        self.auto = Control("☐ Play on select", "toggle_auto", classes="control auto")
        self.loop = Control("↻ Loop", "toggle_loop", classes="control loop optional")
        self.volume = Static("", classes="volume optional")
        self.view = Control("", "next_mode", classes="control view")
        self._shown: tuple[str, str] = ("", "")

    def compose(self) -> ComposeResult:
        yield self.play
        yield self.stop
        yield self.time
        yield self.auto
        yield self.loop
        yield Control("−", "volume(-0.05)", classes="control vol-down optional")
        yield self.volume
        yield Control("+", "volume(0.05)", classes="control vol-up optional")
        yield self.view

    def show(self, state: TransportState) -> None:
        self.play.show("❚❚ Pause" if state.playing else "▶ Play", lit=state.playing)
        self.stop.show("■ Stop", enabled=state.playing)
        self.auto.show(f"{'☑' if state.play_on_select else '☐'} Play on select", lit=state.play_on_select)
        self.loop.show("↻ Loop", lit=state.loop, enabled=state.loop)
        self.view.show(f"{state.view} ▸", lit=True)
        time, volume = f"{state.position} / {state.duration}", f"{state.volume}%"
        if (time, volume) != self._shown:
            self._shown = (time, volume)
            self.time.update(time)
            self.volume.update(volume)
