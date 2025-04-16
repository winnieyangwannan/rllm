import gradio as gr
from PIL import Image
import torch
from fire import Fire

def main(save_path: str):
    """
    Load trajectory data from the given save path and launch a Gradio interface
    for visualizing different trajectories.
    
    Args:
        save_path (str): Path to the directory where 'evaluate_trajectories.pt' is stored.
    """
    # Load the saved trajectories using PyTorch
    trajs = torch.load(f"{save_path}/evaluate_trajectories.pt")
    # Remove any empty trajectories from the list
    trajs = list(filter(lambda x: len(x) > 0, trajs))

    def update_label(i: int):
        """
        For a given trajectory index, extract the task, actions, and images and
        combine the images into one final image.
        
        Args:
            i (int): Index of trajectory to display.
            
        Returns:
            tuple: (task, formatted action sequence, final answer, reference,
                    evaluation info, reward, combined image)
        """
        # Extract the task or goal from the first step's observation
        task = trajs[i][0]["observation"]['goal']
        
        images = []  # List to store PIL Image objects for each step
        
        # Process each step in the selected trajectory
        for step in trajs[i]:
            # Extract the screenshot (assumed to be a numpy array) from the observation
            img_array = step['observation']['screenshot']
            # Convert the numpy array to a PIL Image and ensure it's in RGB format
            img = Image.fromarray(img_array).convert("RGB")
            images.append(img)
        
        # Combine images into a single image, arranged in rows (1 image per row)
        if images:
            max_images_per_row = 1
            # Determine the number of rows needed (will be equal to the number of images)
            num_rows = (len(images) + max_images_per_row - 1) // max_images_per_row
            
            row_images = []     # List to store combined images for each row
            max_row_width = 0   # Maximum width among all rows
            total_height = 0    # Sum of heights of all rows

            # Combine images row by row
            for row in range(num_rows):
                start_index = row * max_images_per_row
                end_index = min(start_index + max_images_per_row, len(images))
                row_images_subset = images[start_index:end_index]

                # Calculate total width of the current row and its maximum height
                total_width = sum(img.width for img in row_images_subset)
                max_height = max(img.height for img in row_images_subset)

                # Create a blank image to hold the current row's images
                row_image = Image.new('RGB', (total_width, max_height))
                x_offset = 0
                for img in row_images_subset:
                    row_image.paste(img, (x_offset, 0))
                    x_offset += img.width

                row_images.append(row_image)
                max_row_width = max(max_row_width, total_width)
                total_height += max_height

            # Merge all row images vertically into one combined image
            combined_image = Image.new('RGB', (max_row_width, total_height))
            y_offset = 0
            for row_img in row_images:
                combined_image.paste(row_img, (0, y_offset))
                y_offset += row_img.height

        # Extract the final "action" from the last step (as an answer)
        answer = trajs[i][-1]["action"]
        # Build a string that lists all actions for each step in the trajectory
        actions = ""
        for j, step in enumerate(trajs[i]):
            actions += f"{j}: {step['action']}\n"

        # Retrieve reference and evaluation info if available; otherwise, use default messages
        reference = trajs[i][-1].get("reference", "No reference")
        evaluation = trajs[i][-1].get("eval_info", "No evaluation info")
        reward = str(trajs[i][-1]["reward"])

        return task, actions, answer, reference, evaluation, reward, combined_image

    # Define the Gradio interface with a number input and outputs for text and images
    interface = gr.Interface(
        fn=update_label,
        inputs=[gr.components.Number(label="Index")],
        outputs=[
            gr.components.Text(label="Task"),
            gr.components.Text(label="Actions"),
            gr.components.Text(label="Answer"),
            gr.components.Text(label="Reference"),
            gr.components.Text(label="Evaluation"),
            gr.components.Text(label="Reward"),
            gr.components.Image(label="Images")
        ],
        title="Trajectory Visualizer",
        description="Change the index to see different trajectories."
    )

    # Launch the interface with sharing enabled on port 9781
    interface.launch(share=True, server_port=9781)

if __name__ == "__main__":
    Fire(main)
