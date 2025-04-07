from import_lib import *
from torch import nn

def compute_accuracy(model, dataloader, device="cpu"):
    """Compute accuracy."""
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss().to(device)

    loss_collector = []
 
    with torch.no_grad():
        for _, (x, target) in enumerate(dataloader):

            x = x.to(device)
            target = target.to(dtype=torch.int64, device=device)
            _, _, out = model(x)
            
            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)
            loss_collector.append(loss.item())
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

        avg_loss = sum(loss_collector) / len(loss_collector)
   

    if was_training:
        model.train()

    return correct / float(total), avg_loss

def train_moon(
    net,
    global_net,
    previous_net,
    train_dataloader,
    lr,
    temperature,
    device="cpu",
    epochs=1,
    mu=0.005,
):
    """Training function for MOON."""
    net.to(device)
    global_net.to(device)
    previous_net.to(device)

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-5,
    )

    criterion = nn.CrossEntropyLoss().cuda()

    previous_net.eval()
    for param in previous_net.parameters():
        param.requires_grad = False

    cos = torch.nn.CosineSimilarity(dim=-1)

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []

        for _, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            # pro1 is the representation by the current model (Line 14 of Algorithm 1)
            _, pro1, out = net(x)
            # pro2 is the representation by the global model (Line 15 of Algorithm 1)
            _, pro2, _ = global_net(x)
            # posi is the positive pair
            posi = cos(pro1, pro2)
            logits = posi.reshape(-1, 1)

            # pro 3 is the representation by the previous model (Line 16 of Algorithm 1)
            _, pro3, _ = previous_net(x)
            # nega is the negative pair
            nega = cos(pro1, pro3)
            logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1) / temperature

            labels = torch.zeros(x.size(0), device=device).long()
            # compute the model-contrastive loss (Line 17 of Algorithm 1)
            loss2 = mu * criterion(logits, labels)
            # compute the cross-entropy loss (Line 13 of Algorithm 1)
            loss1 = criterion(out, target)
            # compute the loss (Line 18 of Algorithm 1)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        print(f"Epoch: {epoch} - Loss: {epoch_loss:.4f} - Loss1: {epoch_loss1:.4f} - Loss2: {epoch_loss2:.4f}")

    previous_net.to("cpu")
    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    net.to("cpu")
    global_net.to("cpu")
    print(f">> Training accuracy: {train_acc:.6f}")
    print(" ** Training complete **")
    return net, epoch_loss, train_acc

def test_moon(net, test_dataloader, device="cpu"):
    """Test function."""
    net.to(device)
    test_acc, loss = compute_accuracy(net, test_dataloader, device=device)
    print(f">> Test accuracy: {test_acc:.6f}")
    net.to("cpu")
    return test_acc, loss