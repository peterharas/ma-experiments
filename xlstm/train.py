import torch
from tqdm import tqdm

from ray import tune

def train_model(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    device,
    epochs,
    patience,
    model_save_path=None
):
    print(f"Device: {device}")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in epochs:

        # -------------------------
        # TRAINING
        # -------------------------

        model.train()

        train_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}")

        for xb, yb in train_bar:

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

        # -------------------------
        # VALIDATION
        # -------------------------

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

        # -------------------------
        # EARLY STOPPING
        # -------------------------

        tune.report(
            val_loss=val_loss,
            epoch=epoch
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            patience_counter = 0

            if model_save_path is not None:
                torch.save(
                    model.state_dict(),
                    model_save_path
                )
                print("    Best model updated")

        else:

            patience_counter += 1

            print(
                f"    No improvement "
                f"({patience_counter}/{patience})"
            )

            if patience_counter >= patience:

                print("Early stopping triggered")

                break

    return best_val_loss
