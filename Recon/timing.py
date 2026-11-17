"""Small wall-clock stage timer; optional sync accounts for asynchronous devices."""
from time import perf_counter


class StageTimer:
    def __init__(self, synchronize=None):
        self.synchronize = synchronize
        if synchronize is not None:
            synchronize()
        self.started = self.previous = perf_counter()
        self.timings = {}

    def mark(self, name):
        if self.synchronize is not None:
            self.synchronize()
        now = perf_counter()
        self.timings[name] = (now - self.previous) * 1000.
        self.previous = now

    def finish(self):
        self.timings['total_ms'] = (perf_counter() - self.started) * 1000.
        return dict(self.timings)


def timing_summary(timings):
    """All values are milliseconds; total is not an additional stage."""
    total = timings.get('total_ms', 0.)
    stages = ((key, value) for key, value in timings.items() if key != 'total_ms')
    return f'总计 {total:.1f} ms; ' + '; '.join(
        f'{key}={value:.1f} ms' for key, value in sorted(stages, key=lambda item: -item[1]))
