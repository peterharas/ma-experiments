import torch
from tqdm import tqdm
from ray import train as ray_train # Import modern Ray Train API

def train_model(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    device,
    epochs,
    patience,
    model_save_path=None,
    use_ray=False,
    verbose=True
):
    print(f"Device: {device}")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0
        iterable = tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}") if verbose else train_loader

        for xb, yb in iterable:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)

            loss = criterion(preds, yb)
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():

            for xb, yb in valid_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item()

        val_loss /= len(valid_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| train_loss={train_loss:.5f} "
            f"| val_loss={val_loss:.5f}"
        )

        if use_ray:
            ray_train.report({
                "val_loss": val_loss,
                "train_loss": train_loss,
                "epoch": epoch + 1
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            if model_save_path is not None:
                torch.save(model.state_dict(), model_save_path)
                print("    Best model updated")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return best_val_loss
