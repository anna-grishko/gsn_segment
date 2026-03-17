import os
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
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
        master.title("Sex prediction based on GSN shape")
        master.geometry("500x450")  # Slightly taller to fit result labels
        self.folder_path = ""

        # --- Folder Selection ---
        self.path_label = tk.Label(master, text="No folder selected", wraplength=450, fg="gray")
        self.path_label.pack(pady=10)

        self.browse_button = tk.Button(master, text="Select Input Folder", command=self.browse_folder)
        self.browse_button.pack(pady=5)

        self.process_button = tk.Button(master, text="Process All & Export CSV", command=self.run_batch_prediction,
                                        state=tk.DISABLED, bg='#28a745', fg='white', font=('Arial', 10, 'bold'))
        self.process_button.pack(pady=15)

        # --- Status and Progress ---
        self.status_label = tk.Label(master, text="Status: Ready", font=('Arial', 10, 'italic'))
        self.status_label.pack(pady=5)

        # --- Dynamic Results Display Section ---
        tk.Label(master, text="--- Current Image Results ---", font=('Arial', 10, 'bold')).pack(pady=(10, 0))

        self.res_file_label = tk.Label(master, text="File: -", fg="blue")
        self.res_file_label.pack()

        self.res_pred_label = tk.Label(master, text="Predicted Sex: -", font=('Arial', 12, 'bold'))
        self.res_pred_label.pack()

        self.res_prob_m_label = tk.Label(master, text="Prob Male: -")
        self.res_prob_m_label.pack()

        self.res_prob_f_label = tk.Label(master, text="Prob Female: -")
        self.res_prob_f_label.pack()

    def browse_folder(self):
        self.folder_path = filedialog.askdirectory()
        if self.folder_path:
            self.path_label.config(text=f"Folder: {self.folder_path}", fg="black")
            self.process_button.config(state=tk.NORMAL)

    def run_batch_prediction(self):
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')
        files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(valid_extensions)]

        if not files:
            messagebox.showwarning("No Images", "No valid images found in the selected folder.")
            return

        self.process_button.config(state=tk.DISABLED, text="Processing...")
        results_data = []

        try:
            scaler = joblib.load(SCALER_PATH)
            classifier = joblib.load(CLASSIFIER_PATH)

            for i, file_name in enumerate(files):
                full_path = os.path.join(self.folder_path, file_name)

                # Update Status
                self.status_label.config(text=f"Processing {i + 1} of {len(files)}")
                self.master.update_idletasks()

                # Get results
                pred, prob_m, prob_f = process_image_and_predict(full_path, scaler, classifier)

                # Update the App Display for the current image
                self.res_file_label.config(text=f"File: {file_name}")
                self.res_pred_label.config(text=f"Predicted Sex: {pred}",
                                           fg="#0056b3" if pred == "Female" else "#d9534f")
                self.res_prob_m_label.config(text=f"Prob Male: {prob_m:.4f}")
                self.res_prob_f_label.config(text=f"Prob Female: {prob_f:.4f}")

                # IMPORTANT: This forces the GUI to redraw and show the labels above
                self.master.update()

                results_data.append({
                    'filename': file_name,
                    'predicted_sex': pred,
                    'probability_male': f"{prob_m:.4f}",
                    'probability_female': f"{prob_f:.4f}"
                })

            # Save CSV
            os.makedirs(FOLDER_TO_SAVE_RESULTS, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(FOLDER_TO_SAVE_RESULTS, f"batch_results_{timestamp}.csv")

            with open(csv_path, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['filename', 'predicted_sex', 'probability_male',
                                                       'probability_female'])
                writer.writeheader()
                writer.writerows(results_data)

            messagebox.showinfo("Success", f"Summary saved to:\n{csv_path}")
            self.status_label.config(text="Status: Finished")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

        finally:
            self.process_button.config(state=tk.NORMAL, text="Process All & Export CSV")


if __name__ == "__main__":
    root = tk.Tk()
    app = PredictorApp(root)
    root.mainloop()
