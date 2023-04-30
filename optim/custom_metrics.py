import torch
import torchmetrics

class CustomAccuracy(torchmetrics.Metric):
    def __init__(self, threshold=0.04, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold
        self.add_state("correct", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        assert predictions.shape == targets.shape, "Predictions and targets must have the same shape"
        abs_diff = torch.abs(predictions - targets)
        correct = torch.sum(abs_diff <= self.threshold).float()
        self.correct += correct
        self.total += targets.numel()

    def compute(self):
        accuracy = self.correct / self.total
        return accuracy

# Example usage
# predictions = torch.tensor([0.11, 0.25, 0.33, 0.48, 0.52])
# targets = torch.tensor([0.1, 0.24, 0.55, 0.45, 0.55])
# metric = CustomAccuracy(threshold=0.04)
# metric.update(predictions, targets)
# accuracy = metric.compute()

# print("Custom accuracy:", accuracy.item())
