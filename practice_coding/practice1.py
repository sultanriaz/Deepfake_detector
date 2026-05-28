import torch

x=torch.randn(100,1)
y=5*x-1
w=torch.randn(1, requires_grad=True)
b=torch.randn(1, requires_grad=True)
learning_rate=0.1
epochs=10
for i in range(epochs):
    y_pred=w*x+b
    loss= ((y_pred-y)**2).mean()
    loss.backward()
    with torch.no_grad():
        w-=learning_rate*w.grad
        b-=learning_rate*b.grad
    w.grad.zero_()
    b.grad.zero_()
    print(f"Epoch {i}, Loss: {loss.item()}")