import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_vertical_architecture():
    if not os.path.exists("docs"):
        os.makedirs("docs")
        
    fig, ax = plt.subplots(figsize=(7, 13))
    ax.axis('off')

    # Define standard box dimensions
    box_w = 4.0
    box_h = 1.0
    
    # Centers
    x_center = 5
    y_start = 12
    y_step = 1.6
    
    blocks = [
        "Multivariate Time Series Input\n(Sensor Data)",
        "Preprocessing & Scaling",
        "Autoencoder\n(Feature Extraction)",
        "Stacked LSTM\n(Sequence Learning)",
        "Attention Mechanism\n(Temporal Weighting)",
        "Regression Head\n(Dense Layers)",
        "Predicted AQI\n(CO(GT))",
        "Streamlit Web Interface"
    ]
    
    for i, text in enumerate(blocks):
        y = y_start - i * y_step
        
        # Draw main box
        ax.add_patch(patches.Rectangle((x_center - box_w/2, y - box_h/2), box_w, box_h, 
                                       fill=True, color='white', lw=1.5, ec='black'))
        ax.text(x_center, y, text, ha='center', va='center', fontsize=10, fontweight='bold', wrap=True)
        
        # Draw arrow downwards to next block
        if i < len(blocks) - 1:
            next_y = y_start - (i + 1) * y_step
            ax.annotate('', xy=(x_center, next_y + box_h/2), xytext=(x_center, y - box_h/2),
                        arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
            
    # Add a side branch for Attention Weights (similar to Grad-CAM Heatmap)
    # The side branch connects from Stacked LSTM (index 3)
    lstm_y = y_start - 3 * y_step
    side_box_w = 2.5
    side_x_center = x_center + box_w/2 + 2
    
    # Draw side box
    ax.add_patch(patches.Rectangle((side_x_center - side_box_w/2, lstm_y - box_h/2), side_box_w, box_h, 
                                   fill=True, color='white', lw=1.5, ec='black'))
    ax.text(side_x_center, lstm_y, "Attention\nWeights", ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw arrow from LSTM to side box
    ax.annotate('', xy=(side_x_center - side_box_w/2, lstm_y), xytext=(x_center + box_w/2, lstm_y),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
                
    # Also an arrow back to Attention Mechanism from Attention Weights? No, just an output side branch to visualize.

    plt.xlim(1, 10)
    plt.ylim(0, 13)
    # plt.title("Proposed Model Architecture", fontsize=14, fontweight='bold')
    # IEEE papers usually have captions below the figure instead of titles inside.
    plt.tight_layout()
    plt.savefig('docs/architecture_flowchart.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    print("Flowchart saved to docs/architecture_flowchart.png")

if __name__ == "__main__":
    draw_vertical_architecture()
