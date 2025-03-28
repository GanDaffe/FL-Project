from import_lib import *
from torch import nn

def compute_accuracy(model, dataloader, device="cpu", multiloader=False):
    """Compute accuracy."""
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    if device == "cpu":
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []
    if multiloader:
        for loader in dataloader:
            with torch.no_grad():
                for _, (x, target) in enumerate(loader):
                    if device != "cpu":
                        x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                    _, _, out = model(x)
                    if len(target) == 1:
                        loss = criterion(out, target)
                    else:
                        loss = criterion(out, target)
                    _, pred_label = torch.max(out.data, 1)
                    loss_collector.append(loss.item())
                    total += x.data.size()[0]
                    correct += (pred_label == target.data).sum().item()

                    if device == "cpu":
                        pred_labels_list = np.append(
                            pred_labels_list, pred_label.numpy()
                        )
                        true_labels_list = np.append(
                            true_labels_list, target.data.numpy()
                        )
                    else:
                        pred_labels_list = np.append(
                            pred_labels_list, pred_label.cpu().numpy()
                        )
                        true_labels_list = np.append(
                            true_labels_list, target.data.cpu().numpy()
                        )
        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        with torch.no_grad():
            for _, (x, target) in enumerate(dataloader):
                # print("x:",x)
                if device != "cpu":
                    x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                _, _, out = model(x)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss.item())
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(
                        pred_labels_list, pred_label.cpu().numpy()
                    )
                    true_labels_list = np.append(
                        true_labels_list, target.data.cpu().numpy()
                    )
            avg_loss = sum(loss_collector) / len(loss_collector)

    if was_training:
        model.train()

    return correct / float(total), avg_loss

def train_moon(
    net,
    global_net,
    previous_net,
    train_dataloader,
    epochs,
    lr,
    mu,
    temperature,
    device="cpu",
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
    previous_net.cuda()

    cnt = 0
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

            previous_net.to(device)
            # pro 3 is the representation by the previous model (Line 16 of Algorithm 1)
            _, pro3, _ = previous_net(x)
            # nega is the negative pair
            nega = cos(pro1, pro3)
            logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

            previous_net.to("cpu")
            logits /= temperature
            labels = torch.zeros(x.size(0)).cuda().long()
            # compute the model-contrastive loss (Line 17 of Algorithm 1)
            loss2 = mu * criterion(logits, labels)
            # compute the cross-entropy loss (Line 13 of Algorithm 1)
            loss1 = criterion(out, target)
            # compute the loss (Line 18 of Algorithm 1)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        print(
            "Epoch: %d Loss: %f Loss1: %f Loss2: %f"
            % (epoch, epoch_loss, epoch_loss1, epoch_loss2)
        )

    previous_net.to("cpu")
    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)

    print(">> Training accuracy: %f" % train_acc)
    net.to("cpu")
    global_net.to("cpu")
    print(" ** Training complete **")
    return net, epoch_loss, train_acc

def test_moon(net, test_dataloader, device="cpu"):
    """Test function."""
    net.to(device)
    test_acc, loss = compute_accuracy(net, test_dataloader, device=device)
    print(">> Test accuracy: %f" % test_acc)
    net.to("cpu")
    return test_acc, loss