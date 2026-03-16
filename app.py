import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import joblib
import numpy as np
from matplotlib import pyplot as plt

# --- Import custom modules/functions ---
from utils.curve_fitting_utils import sample_equidist
from utils.ef_functions import ef_coefs
from utils.img_preproc_utils import ImgProcessor, ContourGSN, transform_gsn



# Configuration and Paths (Adjust as needed)
FOLDER_TO_SAVE_RESULTS = 'data/segmented_curves_gui_output'
SCALER_PATH = 'utils/scaler.save'
CLASSIFIER_PATH = 'utils/the_model.save'


# --- Main Processing Logic (Adapted from your script) ---
def process_image_and_predict(file_path, scaler, classifier):
    """
    Runs the full image processing, segmentation, and prediction pipeline.
    """
    if not file_path:
        raise ValueError("No file path provided.")

    folder, file_name = os.path.split(file_path)

    # Check for model/scaler files (essential for prediction)
    if not os.path.exists(SCALER_PATH) or not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(f"Missing required files: '{SCALER_PATH}' or '{CLASSIFIER_PATH}'. "
                                "Ensure they are in the execution directory.")

    print(f'Processing image: {file_name}')

    # 1. Image Read and Preprocessing
    img = ImgProcessor.read(folder, file_name)
    img_new = img.resize()
    masked_img = img_new.mask()


    # 2. Contour Finding and Transformation
    img_contour = ContourGSN.find_contour(img_new, masked_img['mask'])
    img_contour.segment_gsn()
    res = transform_gsn(img_contour.flipped_data['points'][0],
                        img_contour.flipped_data['points'][1],
                        img_contour.flipped_data['curve'])

    # 3. Save Plots and Intermediate Results
    new_subfolder = os.path.splitext(file_name)[0]
    output_dir = os.path.join(FOLDER_TO_SAVE_RESULTS, new_subfolder)
    os.makedirs(output_dir, exist_ok=True)

    # Save image and data files
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(cv2.cvtColor(img_new.img, cv2.COLOR_BGR2RGB), cmap=plt.cm.Spectral)
    for contour in img_contour.contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=3, markersize=12, zorder=1)
    ax.scatter(img_contour.points[0][1], img_contour.points[0][0], s=120, marker='X', color='red', zorder=2)
    ax.scatter(img_contour.points[1][1], img_contour.points[1][0], s=120, marker='X', color='red', zorder=3)
    ax.annotate("A", (img_contour.points[0][1], img_contour.points[0][0]), xytext=(10, 10), textcoords='offset points',
                color='red', fontsize=50)
    ax.annotate("B", (img_contour.points[1][1], img_contour.points[1][0]), xytext=(10, 10), textcoords='offset points',
                color='red', fontsize=50)
    plt.savefig(os.path.join(output_dir, os.path.splitext(file_name)[0] + '_segmented.png'),
                bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, os.path.splitext(file_name)[0] + '_segmented.eps'),
                format="eps", dpi=100, bbox_inches="tight", transparent=True)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(res[:, 1], res[:, 0], '-r', linewidth=3)
    ax.set_box_aspect(masked_img['mask'].shape[0] / masked_img['mask'].shape[1])
    ax.set_ylim(ymin=0)
    ax.axis('equal')
    plt.savefig(os.path.join(output_dir, os.path.splitext(file_name)[0] + '_registered.png'),
                bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, os.path.splitext(file_name)[0] + '_registered.eps'),
                format="eps", dpi=600, bbox_inches="tight", transparent=True)
    plt.close()

    np.savetxt(os.path.join(output_dir, 'gsn_curve_registered.csv'), res, delimiter=",")


    # 4. Feature Extraction and Prediction
    coordinates = sample_equidist(res, 100)
    if coordinates[0][1] > coordinates[-1][1]:
        coordinates = coordinates[::-1]
    new_dat = np.array([np.zeros(coordinates.shape[0]), coordinates[::-1, 1]]).T
    new_dat = new_dat[1:(coordinates.shape[0] - 2), :]
    outline = np.concatenate((coordinates, new_dat), axis=0)

    coefs = ef_coefs(outline, n_harmonics=7)

    # scaler = joblib.load(SCALER_PATH)
    # classifier = joblib.load(CLASSIFIER_PATH)

    # Prediction
    X = scaler.transform(coefs.reshape(1, -1))
    y_pred = classifier.predict(X)[0]
    y_prob_M = classifier.predict_proba(X)[0, 0]
    y_prob_F = classifier.predict_proba(X)[0, 1]

    # Return results
    return "Female" if y_pred == 1 else "Male", y_prob_M, y_prob_F


# --- Tkinter GUI Application ---

class PredictorApp:
    def __init__(self, master):
        self.master = master
        master.title("Biological sex prediction based on the GSN shape")
        master.geometry("500x300")

        self.file_path = ""

        # --- GUI Elements ---

        # 1. File Path Label
        self.path_label = tk.Label(master, text="No file selected", wraplength=450)
        self.path_label.pack(pady=10)

        # 2. Browse Button
        self.browse_button = tk.Button(master, text="Select Image File", command=self.browse_file)
        self.browse_button.pack(pady=5)

        # 3. Process Button
        self.process_button = tk.Button(master, text="Run Prediction", command=self.run_prediction,
                                        state=tk.DISABLED, bg='green', fg='white')
        self.process_button.pack(pady=15)

        # 4. Results Section
        self.result_title = tk.Label(master, text="--- Prediction Results ---", font=('Arial', 10, 'bold'))
        self.result_title.pack()

        self.pred_label = tk.Label(master, text="Predicted Sex: N/A", font=('Arial', 12))
        self.pred_label.pack()

        self.prob_m_label = tk.Label(master, text="Probability Male: N/A")
        self.prob_m_label.pack()

        self.prob_f_label = tk.Label(master, text="Probability Female: N/A")
        self.prob_f_label.pack()

    def browse_file(self):
        """Opens a file dialog to select an image."""
        self.file_path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*"))
        )
        if self.file_path:
            self.path_label.config(text=f"Selected File: {os.path.basename(self.file_path)}")
            self.process_button.config(state=tk.NORMAL)
            # Clear previous results
            self.pred_label.config(text="Predicted Sex: Waiting...")
            self.prob_m_label.config(text="Probability Male: N/A")
            self.prob_f_label.config(text="Probability Female: N/A")
        else:
            self.path_label.config(text="No file selected")
            self.process_button.config(state=tk.DISABLED)

    def run_prediction(self):
        """Triggers the image processing and displays the results."""
        if not self.file_path:
            messagebox.showerror("Error", "Please select an image file first.")
            return

        # Disable button while processing to prevent multiple clicks
        self.process_button.config(state=tk.DISABLED, text="Processing...")

        try:
            scaler = joblib.load(SCALER_PATH)
            classifier = joblib.load(CLASSIFIER_PATH)
            # Run the core logic
            y_pred, y_prob_M, y_prob_F = process_image_and_predict(self.file_path, scaler, classifier)
            # Update GUI labels with results
            self.pred_label.config(text=f"Predicted Sex: **{y_pred}**",
                                   font=('Arial', 14, 'bold'), fg='blue')
            self.prob_m_label.config(text=f"Probability Male: {y_prob_M:.4f}")
            self.prob_f_label.config(text=f"Probability Female: {y_prob_F:.4f}")

            print("Prediction complete.")

        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred during processing: {e}")
            self.pred_label.config(text="Predicted Sex: ERROR", fg='red')
            self.prob_m_label.config(text="Probability Male: N/A")
            self.prob_f_label.config(text="Probability Female: N/A")

        finally:
            # Re-enable button
            self.process_button.config(state=tk.NORMAL, text="Run Prediction", bg='green')


# --- Run the App ---
if __name__ == "__main__":
    root = tk.Tk()
    app = PredictorApp(root)
    root.mainloop()
