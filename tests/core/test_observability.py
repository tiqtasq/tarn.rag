"""Observability ABC + NoOpObservability."""

from tarnrag.core.engine.observability import NoOpObservability, Observability


async def test_noop_methods_are_callable_and_do_nothing():
    obs = NoOpObservability()
    obs.counter("a")
    obs.counter("a", 5, tags={"k": "v"})
    obs.gauge("g", 1.5)
    await obs.log("info", "hello", x=1)  # async, no-op


class _RecordingGauges(NoOpObservability):
    """No-op except it records gauge calls — to prove the concrete timer emits one."""

    def __init__(self):
        self.gauges: list[tuple[str, float]] = []

    def gauge(self, name, value, tags=None):
        self.gauges.append((name, value))


def test_timer_emits_a_seconds_gauge():
    obs = _RecordingGauges()
    with obs.timer("stage.Embed.process"):
        pass
    assert len(obs.gauges) == 1
    name, value = obs.gauges[0]
    assert name == "stage.Embed.process.seconds"
    assert value >= 0.0


def test_timer_is_inherited_concrete_on_the_abc():
    # timer is defined on the ABC (not abstract), so every adapter gets it for free.
    assert "timer" not in Observability.__abstractmethods__
    assert Observability.__abstractmethods__ == {"log", "counter", "gauge"}
