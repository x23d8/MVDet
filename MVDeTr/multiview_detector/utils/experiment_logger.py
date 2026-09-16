"""Load W&B only when a run explicitly requests remote logging."""


class _DisabledRun:
    def define_metric(self, *args, **kwargs):
        pass

    def finish(self):
        pass


class _WandbProxy:
    def __init__(self):
        self.module = None

    @property
    def run(self):
        return None if self.module is None else self.module.run

    def init(self, *args, mode='disabled', **kwargs):
        if mode == 'disabled':
            return _DisabledRun()
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError('W&B is unavailable; use --wandb_mode disabled or repair its installation') from exc
        self.module = wandb
        return wandb.init(*args, mode=mode, **kwargs)

    def log(self, metrics):
        if self.run is not None:
            self.module.log(metrics)


wandb = _WandbProxy()
