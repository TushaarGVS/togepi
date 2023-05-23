from torch.optim.lr_scheduler import LRScheduler


class NoamLrScheduler(LRScheduler):
    def __init__(self, optim, warmup_steps, d_model, factor=1.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.factor = factor
        super().__init__(optim, last_epoch)

    def get_lr(self):
        step_num = max(self.last_epoch, 1)
        scale = self.factor * min(step_num ** (-0.5), step_num * self.warmup_steps ** (-1.5))
        return [scale * self.d_model ** 0.5 for _ in self.base_lrs]
