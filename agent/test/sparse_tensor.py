import torch


i = torch.LongTensor([[0, 1, 1],
                          [2, 0, 2]])

v = torch.FloatTensor([3, 4, 5])

torch.sparse.FloatTensor(i, v, torch.Size([2,3])).to_dense()



a = torch.randn(2,3).to_sparse().requires_grad_()
a
b = torch.randn(3,2,requires_grad=True)
b

y = torch.sparse.mm(a,b)
y = torch.nn.Softmax(dim=-1)(y)
y.sum().backward()
a.grad

S = torch.sparse_coo_tensor(indices = torch.tensor([[0,0,1,2],[2,3,0,3]]), values = torch.tensor([1,2,1,3]), size=[3,4])