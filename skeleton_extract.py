from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
import numpy as np
import os
import matplotlib.pyplot as plt

register_all_modules()

# Initialize the model, pre-trained MMPOSE network
config_file = 'PoseDectModel/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
checkpoint_file = 'PoseDectModel/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
model = init_model(config_file, checkpoint_file, device='cpu')  # Or use device='cuda:0'

# Set the folder paths and save path
img_folder = '/root/autodl-tmp/testinstall/Desk/rgb'
save_folder = '/root/autodl-tmp/testinstall/Desk/rgb_pose'
os.makedirs(save_folder, exist_ok=True)

# Define keypoint connections, different parts with different colors
body_parts = {
    'legs': {
        'connections': [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12]],
        'color': 'green'
    },
    'torso': {
        'connections': [[5, 11], [6, 12], [5, 6]],
        'color': 'red'
    },
    'arms': {
        'connections': [[5, 7], [7, 9], [6, 8], [8, 10]],
        'color': 'blue'
    },
    'head': {
        'connections': [[0, 1], [1, 2], [0, 2], [1, 5], [2, 6]],  # Added head to torso connections
        'color': 'purple'
    }
}

# Iterate through images
for img_file in os.listdir(img_folder):
    img_path = os.path.join(img_folder, img_file)
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        results = inference_topdown(model, img_path)
        keypoints = results[0].pred_instances.keypoints[0]

        # Adjust coordinates for plotting needs
        keypoints[:, 1] = np.max(keypoints[:, 1]) - keypoints[:, 1]

        # Set large figure display and background color
        plt.rcParams['figure.figsize'] = [5, 11]
        fig, ax = plt.subplots()
        ax.patch.set_visible(False)  # Ensure axes transparency
        ax.axis('off')  # Turn off axes

        # Plot keypoints and connections
        for part, details in body_parts.items():
            for connection in details['connections']:
                plt.plot([keypoints[connection[0], 0], keypoints[connection[1], 0]],
                         [keypoints[connection[0], 1], keypoints[connection[1], 1]],
                         color=details['color'], alpha=0.6, linewidth=10)  # Adjust transparency and line width

            # Plot keypoints
            for point_index in np.unique(np.ravel(details['connections'])):
                plt.scatter(keypoints[point_index, 0], keypoints[point_index, 1],
                            c=details['color'], s=300)  # Adjust keypoint size

        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.savefig(os.path.join(save_folder, f'skeleton_{img_file}'), bbox_inches='tight', pad_inches=0, transparent=True)  # Save with transparent background
        plt.close()
