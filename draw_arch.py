import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_architecture():
    if not os.path.exists("docs"):
        os.makedirs("docs")
        
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # Draw blocks
    blocks = [
        ("Input Data\n(Multivariate Time Series)", 1, 4),
        ("Autoencoder (Encoder)\n(Module 2)", 3, 4),
        ("LSTM Layer\n(Module 1)", 5, 4),
        ("Attention Mechanism\n(Module 2)", 7, 4),
        ("Dense Layers\n& Output (AQI)", 9, 4)
    ]

    for i, (text, x, y) in enumerate(blocks):
        # Draw box
        ax.add_patch(patches.Rectangle((x-0.8, y-0.6), 1.6, 1.2, fill=True, color='lightblue', lw=2, ec='black'))
        # Add text
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Draw arrow
        if i < len(blocks) - 1:
            ax.annotate('', xy=(x+0.8, y), xytext=(blocks[i+1][1]-0.8, y),
                        arrowprops=dict(arrowstyle='<-', lw=2))

    plt.title("Model Architecture Diagram", fontsize=14, fontweight='bold', y=0.85)
    plt.tight_layout()
    plt.savefig('docs/architecture.png', dpi=300, bbox_inches='tight')
    print("Architecture diagram saved to docs/architecture.png")

if __name__ == "__main__":
    draw_architecture()
