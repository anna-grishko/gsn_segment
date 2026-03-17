import os
import csv
import joblib
import numpy as np
import cv2
import customtkinter as ctk
from tkinter import filedialog, messagebox
from datetime import datetime
from matplotlib import pyplot as plt

# --- Import your custom modules ---
from utils.curve_fitting_utils import sample_equidist
from utils.ef_functions import ef_coefs
from utils.img_preproc_utils import ImgProcessor, ContourGSN, transform_gsn

# Configuration
FOLDER_TO_SAVE_RESULTS = 'data/segmented_curves_gui_output'
SCALER_PATH = 'utils/scaler.save'
CLASSIFIER_PATH = 'utils/the_model.save'

# Set UI Theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def process_image_and_predict(file_path, scaler, classifier, base_output_folder):
    """Core logic to process image and return prediction results."""
    if not file_path:
        raise ValueError("No file path provided.")

    folder, file_name = os.path.split(file_path)

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

    # 3. Save Plots (Simplified but functional)
    new_subfolder = os.path.splitext(file_name)[0]
    output_dir = os.path.join(base_output_folder, new_subfolder)
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
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(res[:, 1], res[:, 0], '-r', linewidth=3)
    ax.set_box_aspect(masked_img['mask'].shape[0] / masked_img['mask'].shape[1])
    ax.set_ylim(ymin=0)
    ax.axis('equal')
    plt.savefig(os.path.join(output_dir, os.path.splitext(file_name)[0] + '_registered.png'),
                bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, os.path.splitext(file_name)[0] + '_registered.eps'),
                format="eps", dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)  # CRITICAL: Close plot to free memory during batch

    # 4. Feature Extraction
    coordinates = sample_equidist(res, 100)
    if coordinates[0][1] > coordinates[-1][1]:
        coordinates = coordinates[::-1]

    new_dat = np.array([np.zeros(coordinates.shape[0]), coordinates[::-1, 1]]).T
    new_dat = new_dat[1:(coordinates.shape[0] - 2), :]
    outline = np.concatenate((coordinates, new_dat), axis=0)

    coefs = ef_coefs(outline, n_harmonics=7)

    # 5. Prediction
    X = scaler.transform(coefs.reshape(1, -1))
    y_pred = classifier.predict(X)[0]
    y_prob_M = classifier.predict_proba(X)[0, 0]
    y_prob_F = classifier.predict_proba(X)[0, 1]

    pred_text = "Female" if y_pred == 1 else "Male"
    return pred_text, y_prob_M, y_prob_F


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Biological Sex Prediction - GSN Shape")
        self.geometry("600x650")

        self.input_folder = ""
        self.output_folder = ""

        # Title
        self.title_label = ctk.CTkLabel(self, text="GSN Batch Processor", font=("Arial", 24, "bold"))
        self.title_label.pack(pady=(20, 10))

        # --- Input Selection ---
        self.in_path_label = ctk.CTkLabel(self, text="No input folder selected", text_color="gray")
        self.in_path_label.pack()
        self.btn_browse_in = ctk.CTkButton(self, text="Select Input Folder", command=self.browse_input)
        self.btn_browse_in.pack(pady=(5, 15))

        # --- Output Selection ---
        self.out_path_label = ctk.CTkLabel(self, text="No output folder selected", text_color="gray")
        self.out_path_label.pack()
        self.btn_browse_out = ctk.CTkButton(self, text="Select Output Folder", command=self.browse_output)
        self.btn_browse_out.pack(pady=(5, 20))

        # Run Button
        self.btn_run = ctk.CTkButton(self, text="Start Batch Process", state="disabled",
                                     fg_color="#28a745", hover_color="#218838", command=self.run_process)
        self.btn_run.pack(pady=10)

        # Status Display
        self.status_label = ctk.CTkLabel(self, text="Status: Ready", font=("Arial", 12, "italic"))
        self.status_label.pack(pady=5)

        # Result Card
        self.res_frame = ctk.CTkFrame(self)
        self.res_frame.pack(pady=20, padx=40, fill="both", expand=True)

        self.res_file = ctk.CTkLabel(self.res_frame, text="Predicted file: -")
        self.res_file.pack(pady=10)
        self.res_pred = ctk.CTkLabel(self.res_frame, text="Predicted Sex: -", font=("Arial", 20, "bold"))
        self.res_pred.pack(pady=5)
        self.res_prob_m = ctk.CTkLabel(self.res_frame, text="Prob Male: -")
        self.res_prob_m.pack()
        self.res_prob_f = ctk.CTkLabel(self.res_frame, text="Prob Female: -")
        self.res_prob_f.pack(pady=(0, 15))

    def check_ready(self):
        """Enables the start button only if both paths are set."""
        if self.input_folder and self.output_folder:
            self.btn_run.configure(state="normal")

    def browse_input(self):
        self.input_folder = filedialog.askdirectory(title="Select Input Folder")
        if self.input_folder:
            self.in_path_label.configure(text=f"Input: {self.input_folder}", text_color="white")
            self.check_ready()

    def browse_output(self):
        self.output_folder = filedialog.askdirectory(title="Select Output Folder")
        if self.output_folder:
            self.out_path_label.configure(text=f"Output: {self.output_folder}", text_color="white")
            self.check_ready()

    def run_process(self):
        valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')
        files = [f for f in os.listdir(self.input_folder) if f.lower().endswith(valid_ext)]

        if not files:
            messagebox.showwarning("No Images", "No valid image files found in the input folder.")
            return

        # 2. Setup
        self.btn_run.configure(state="disabled", text="Processing...")
        results_data = []

        try:
            scaler = joblib.load(SCALER_PATH)
            classifier = joblib.load(CLASSIFIER_PATH)

            for i, file_name in enumerate(files):
                full_path = os.path.join(self.input_folder, file_name)

                self.status_label.configure(text=f"Processing {i + 1} of {len(files)}")
                self.res_file.configure(text=f"Predicted file: {file_name}")
                # self.update()

                # Predict (Now passing the output folder)
                pred, prob_m, prob_f = process_image_and_predict(full_path, scaler, classifier, self.output_folder)

                color = "#5dade2" if pred == "Female" else "#ec7063"
                self.res_pred.configure(text=f"Result: {pred}", text_color=color)
                self.res_prob_m.configure(text=f"Prob Male: {prob_m:.4f}")
                self.res_prob_f.configure(text=f"Prob Female: {prob_f:.4f}")
                self.update()

                results_data.append({
                    'filename': file_name,
                    'predicted_sex': pred,
                    'probability_male': f"{prob_m:.4f}",
                    'probability_female': f"{prob_f:.4f}"
                })

            # Save CSV in the custom output folder
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(self.output_folder, f"summary_{ts}.csv")

            with open(csv_path, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['filename', 'predicted_sex', 'probability_male',
                                                       'probability_female'])
                writer.writeheader()
                writer.writerows(results_data)

            messagebox.showinfo("Complete",
                                f"Successfully processed {len(files)} files.\n\nSummary saved to:\n{csv_path}")

        except Exception as e:
            messagebox.showerror("Runtime Error", f"An error occurred: {str(e)}")

        finally:
            self.status_label.configure(text="Status: Finished")
            self.btn_run.configure(state="normal", text="Start Batch Process")


if __name__ == "__main__":
    app = App()
    app.mainloop()