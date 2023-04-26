import torch
import torch.nn as nn

# Example of target with class indices
# m = torch.tensor([1., 0., 1., 0., 1.])
# loss = nn.CrossEntropyLoss(weight=m)
# logits = torch.randn(3, 5, requires_grad=True)
# target = torch.LongTensor([1, 2, 3])
# output = loss(logits * m, target)
# output.backward()
#
# print(logits)
# print(logits.grad)


i = torch.randn(1, 5, requires_grad=True)
t = torch.LongTensor([2])
n = nn.Linear(5, 5)
# opt = torch.optim.AdamW(n.parameters(), lr=0.001, amsgrad=True)
opt = torch.optim.RMSprop(n.parameters(), lr=0.0001, momentum=0.9)


def step():
    opt.zero_grad()
    o = n(i)
    m = {
        0: 0,
        2: 1,
        4: 2,
    }
    o_ = o[:, [0, 2, 4]]
    t_ = torch.LongTensor([m[t.item()]])
    l = nn.CrossEntropyLoss()(o_, t_)
    # l = nn.CrossEntropyLoss()(o, t)
    l.backward()
    opt.step()
