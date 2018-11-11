import torch_test
import torch
import torch.nn.functional as F

scale = torch.tensor([1.0, 2.0, 3.0])

print('-----------------------')
print('reference')

i = torch.tensor([[3.0, 4.0, 0], [6, 7, 5]])
i.requires_grad_()
o = F.normalize(i, p=2, dim=1)
print(o)
(o * scale).sum().backward()
print(i.grad)

print('-----------------------')
print('reference 2')

i = torch.tensor([[3.0, 4.0, 0], [6, 7, 5]])
i.requires_grad_()
o = F.normalize(i, p=2, dim=1)
print(o)
F.normalize(o * scale, p=2, dim=1).sum().backward()
print(i.grad)

for k in range(3):
    print('-----------------------')
    i = torch.tensor([[3.0, 4.0, 0], [6, 7, 5]])

    i.requires_grad_()
    o = torch_test.normalize(i)
    print(o)
    (o * scale).sum().backward()
    print(i.grad)


for k in range(3):
    print('-----------------------')
    torch_test.clear_graph()
    i = torch.tensor([[3.0, 4.0, 0], [6, 7, 5]])

    i.requires_grad_()
    o = torch_test.normalize(i)
    print(o)
    (o * scale).sum().backward()
    print(i.grad)

for k in range(3):
    print('-----------------------')
    torch_test.clear_graph()
    i = torch.tensor([[3.0, 4.0, 0], [6, 7, 5]])

    i.requires_grad_()
    o = torch_test.normalize(i)
    print(o)
    torch_test.normalize(o * scale).sum().backward()
    print(i.grad)
