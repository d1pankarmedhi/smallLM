import matplotlib.pyplot as plt
import torch


def estimate_loss(model, get_batch, config):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            X, Y = X.to(config.device), Y.to(config.device)
            with torch.no_grad():
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out


def plot_losses(
    train_losses,
    val_losses,
    eval_interval=500,
    save_path="logs/loss_plot.png",
):
    steps = [i * eval_interval for i in range(len(train_losses))]

    plt.figure(figsize=(6, 5))
    plt.plot(steps, train_losses, label="Train Loss")
    plt.plot(steps, val_losses, label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss over Time")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"📈 Loss curve saved to {save_path}")
    else:
        plt.show()
