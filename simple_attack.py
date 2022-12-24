from typing_extensions import dataclass_transform
import torch
from attack_framework import Attack
import torchvision.transforms as transforms

class simple_attack(Attack):
  def __init__(self, substitute_model):
    """
    Parameters
    ----------
    substitute_model : the model architecture that will be trained by the attacker
    dataloader: dataloader for unlabelled data points to be queried using the victim model
    """

    self.substitute_model = substitute_model

  def _run(self, victim_model, x, **kwargs):
    victim_model.eval()
    return victim_model(torch.unsqueeze(x, 0))

  def train(self, substitute_model, num_epochs, dataset, label_list, optimizer, scheduler, loss_fn, batch_size, **kwargs):
    substitute_model.train()
    trainset = []
    for l in range(len(label_list)):
      trainset.append([torch.squeeze(dataset[l % len(dataset)]), torch.Tensor(label_list[l])])
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    for i in range(num_epochs):
      for x, y in dataloader:
        y_pred = substitute_model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
      if scheduler is not None:
        scheduler.step()

    substitute_model.eval()
    return substitute_model