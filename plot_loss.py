import json
import matplotlib.pyplot as plt
import argparse

def plot_losses(losses_path, val_losses_path=None):
    try:
        with open(losses_path, "r") as f:
            losses = json.load(f)
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku {losses_path}. Najpierw uruchom main.py aby wygenerować logi błędu (loss)!")
        return

    val_losses = None
    if val_losses_path:
        try:
            with open(val_losses_path, "r") as f:
                val_losses = json.load(f)
        except FileNotFoundError:
            print(f"Błąd: Nie znaleziono pliku {val_losses_path}.")

    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='b', linewidth=2, label='Train Loss')
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, marker='s', linestyle='--', color='r', linewidth=2, label='Val Loss')
        
    plt.title('Training and Validation Loss YOLOv1')
    plt.xlabel('Epoka')
    plt.ylabel('Loss (Błąd Treningowy)')
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wizualizacja wykresu straty (Loss) dla YOLOv1")
    parser.add_argument("--losses", type=str, default="train_losses.json", help="Ścieżka do pliku JSON z zapisanymi błędami")
    parser.add_argument("--val_losses", type=str, default=None, help="Ścieżka do pliku JSON z zapisanymi błędami walidacyjnymi")
    
    args = parser.parse_args()
    
    plot_losses(args.losses, args.val_losses)
