import torch
import torch.nn.functional as fn


class TsaCrossEntropyLoss(object):
    def __init__(self,
                 num_steps,
                 num_classes,
                 alpha=5,
                 temperature=5,
                 weight=None,
                 cuda=False,
                 schedule='log',
                 current_step=0,
                 ignore_idx=None):
        if weight is not None:
            self.loss_function = torch.nn.CrossEntropyLoss(
                reduction='none', weight=weight)
        elif ignore_idx:
            self.loss_function = torch.nn.CrossEntropyLoss(
                reduction='none', ignore_index=ignore_idx)
        else:
            self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

        self.num_steps = num_steps
        self.current_step = current_step
        self.num_classes = num_classes
        self.alpha = alpha
        self.cuda = cuda
        self.temperature = temperature
        self.schedule = schedule
        self.const = (1 - 1 / self.num_classes) + 1 / self.num_classes

    def _cal_threshold(self):
        if self.schedule == "linear":
            alpha = torch.tensor([self.current_step / self.num_steps])
        elif self.schedule == "log":
            alpha = 1 - \
                    torch.exp(torch.tensor(
                        [- self.current_step * self.alpha / self.num_steps]))
        elif self.schedule == "exp":
            alpha = torch.exp(torch.Tensor(
                [(self.current_step / self.num_steps - 1) * self.alpha]))
        else:
            raise ValueError("Please provide a schedule in [linear, log, exp]")

        thresh = alpha * self.const
        
        if self.cuda:
            thresh = thresh.to("cuda")
        return thresh

    def _step(self):
        self.current_step += 1

    def _get_mask(self, logits, targets):
        thresh = self._cal_threshold()
        mask = fn.softmax(logits, dim=1).detach()
        mask, pred = torch.max(mask, dim=1, keepdim=False)
        wrong_pred = (torch.abs(pred - targets) > 0)
        mask = (mask < thresh)
        mask |= wrong_pred

        if self.cuda:
            return mask.to("cuda", dtype=torch.float)
        else:
            return mask.to(dtype=torch.float)

    def __call__(self, logits, targets):
        logits = logits / self.temperature

        mask = self._get_mask(logits, targets)
        self.loss_value = self.loss_function(logits, targets) * mask
        self.loss_value = self.loss_value.sum() / torch.max(
            torch.Tensor([mask.sum(), 1]))
        self._step()
        return self.loss_value
