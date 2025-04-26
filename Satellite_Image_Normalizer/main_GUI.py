import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os
import zipfile
import re
from datetime import datetime
from PIL import Image, ImageTk, ImageFilter
from skimage.metrics import structural_similarity as ssim
import customtkinter as ctk

class ImageNormalizerGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Satellite Image Normalizer")
        self.geometry("700x500")
        
        # Configure appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Initialize variables
        self.output_folder = "output"
        self.modified_folder = "modified_images"
        self.competition_folder = "competition_output"
        self.input_folder = None
        self.target_brightness = 150
        self.clip_limit = 2.0
        self.normalized_images = []
        self.reference_hist = None
        self.competition_mode = False
        
        # Main container
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Background image
        try:
            self.bg_image = Image.open("background123.jpg")
            self.blurred_bg = self.bg_image.filter(ImageFilter.GaussianBlur(15))
            self.bg_photo = ImageTk.PhotoImage(self.blurred_bg.resize((900, 700)))
            self.bg_label = tk.Label(self.main_frame, image=self.bg_photo)
            self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception as e:
            print(f"Background image error: {e}")
            self.main_frame.configure(fg_color="#2b2b2b")
        
        # Create widgets
        self.create_widgets()
        
        # Status bar
        self.status = ctk.CTkLabel(self, text="Status: Ready", 
                                 font=("Arial", 11), anchor="w")
        self.status.pack(fill="x", padx=10, pady=5)

    def create_widgets(self):
        """Create the main interface buttons"""
        btn_frame = ctk.CTkFrame(self.main_frame, fg_color="#333333", corner_radius=15)
        btn_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        btn_style = {
            'fg_color': "#4a6ea9",
            'hover_color': "#5b7ec4",
            'text_color': "#ffffff",
            'corner_radius': 25,
            'width': 180,
            'height': 50,
            'font': ("Arial", 14, 'bold')
        }
        
        ctk.CTkButton(
            btn_frame,
            text="SELECT FOLDER",
            command=self.select_input,
            **btn_style
        ).grid(row=0, column=0, padx=15, pady=10)

        ctk.CTkButton(
            btn_frame,
            text="PROCESS ZIP",
            command=self.process_competition_zip,
            **btn_style
        ).grid(row=1, column=0, padx=15, pady=10)

        ctk.CTkButton(
            btn_frame,
            text="NORMALIZE ALL",
            command=self.normalize_all,
            **btn_style
        ).grid(row=0, column=1, padx=15, pady=10)

        ctk.CTkButton(
            btn_frame,
            text="ADJUST SINGLE",
            command=self.adjust_single,
            **btn_style
        ).grid(row=0, column=2, padx=15, pady=10)

        self.comp_mode_var = ctk.StringVar(value="off")
        ctk.CTkSwitch(
            btn_frame,
            text="Global Balanced Mode",
            variable=self.comp_mode_var,
            onvalue="on",
            offvalue="off",
            command=self.toggle_competition_mode,
            font=("Arial", 12)
        ).grid(row=1, column=1, columnspan=2, pady=10)

    def toggle_competition_mode(self):
        """Toggle competition mode (global average vs fixed target)"""
        self.competition_mode = self.comp_mode_var.get() == "on"
        mode = "COMPETITION" if self.competition_mode else "STANDARD"
        self.status.configure(text=f"Mode: {mode} | Target: {self.target_brightness:.1f}")

    def select_input(self):
        """Select input folder with images"""
        self.input_folder = filedialog.askdirectory()
        if self.input_folder:
            self.status.configure(text=f"Selected: {os.path.basename(self.input_folder)}")

    def process_competition_zip(self):
        """Process ZIP file with any number of images"""
        zip_path = filedialog.askopenfilename(filetypes=[("ZIP files", "*.zip")])
        if not zip_path:
            return

        try:
            # Create temp folder for extraction
            temp_folder = "temp_zip_extract"
            os.makedirs(temp_folder, exist_ok=True)
            
            # Extract ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_folder)
            
            # Get all image files (sorted naturally)
            image_files = sorted(
                [f for f in os.listdir(temp_folder) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                key=lambda x: [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', x)]
            )
            
            if not image_files:
                raise ValueError("No images found in ZIP file")
            
            # Set input folder and enable competition mode
            self.input_folder = temp_folder
            self.competition_mode = True
            self.normalize_all(competition_mode=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"ZIP processing failed:\n{str(e)}")
        finally:
            # Clean up temp files
            if os.path.exists(temp_folder):
                for f in os.listdir(temp_folder):
                    os.remove(os.path.join(temp_folder, f))
                os.rmdir(temp_folder)

    def calculate_global_average(self, image_files):
        """Calculate mean brightness across all images"""
        total_sum = 0
        total_pixels = 0
        
        for img_file in image_files:
            img = cv2.imread(os.path.join(self.input_folder, img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                total_sum += np.sum(img)
                total_pixels += img.size
        
        return total_sum / (total_pixels + 1e-10)

    def calculate_perceptual_brightness(self, img):
        """Calculate perceptual brightness considering human vision"""
        if len(img.shape) == 3:  # Color image
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            return np.mean(yuv[:,:,0])
        return np.mean(img)  # Grayscale

    def match_histograms(self, img, reference_hist):
        """Fixed histogram matching with proper integer indices"""
        def safe_interp(cdf, ref_cdf, img_values):
            """Ensure proper integer indices for lookup"""
            min_length = min(len(cdf), len(ref_cdf))
            lookup_table = np.interp(
                cdf[:min_length], 
                ref_cdf[:min_length], 
                np.arange(min_length)
            ).astype(np.uint8)  # Convert to integer type
            return np.clip(lookup_table[img_values], 0, 255)

        if len(img.shape) == 3:
            # Color image - process each channel
            matched = np.zeros_like(img)
            for i in range(3):
                ref_hist = np.array(reference_hist[i], dtype=np.float32)
                
                # Calculate input image histogram
                hist, _ = np.histogram(img[:,:,i].flatten(), 256, [0,256])
                cdf = hist.cumsum()
                cdf_normalized = cdf * ref_hist.max() / cdf.max()
                
                # Calculate reference CDF
                ref_cdf = ref_hist.cumsum()
                ref_cdf_normalized = ref_cdf * cdf.max() / ref_cdf.max()
                
                # Apply with proper integer indices
                matched[:,:,i] = safe_interp(
                    cdf_normalized, 
                    ref_cdf_normalized,
                    img[:,:,i].astype(np.uint8)  # Ensure integer indices
                )
            
            return matched
        
        else:
            # Grayscale image
            ref_hist = np.array(reference_hist, dtype=np.float32)
            
            hist, _ = np.histogram(img.flatten(), 256, [0,256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * ref_hist.max() / cdf.max()
            
            ref_cdf = ref_hist.cumsum()
            ref_cdf_normalized = ref_cdf * cdf.max() / ref_cdf.max()
            
            return safe_interp(
                cdf_normalized,
                ref_cdf_normalized,
                img.astype(np.uint8)  # Ensure integer indices
            )

    def enhanced_normalization(self, img):
        """Enhanced normalization pipeline"""
        img = img.astype(np.float32)
        
        if self.reference_hist is not None:
            img = self.match_histograms(img, self.reference_hist)
        
        current_brightness = self.calculate_perceptual_brightness(img)
        scale_factor = self.target_brightness / (current_brightness + 1e-10)
        normalized = np.clip(img * scale_factor, 0, 255)
        
        clahe = cv2.createCLAHE(
            clipLimit=float(self.clip_limit),
            tileGridSize=(8, 8)
        )
        
        if len(normalized.shape) == 3:
            channels = cv2.split(normalized)
            for i in range(len(channels)):
                channels[i] = clahe.apply(channels[i].astype(np.uint8))
            normalized = cv2.merge(channels)
        else:
            normalized = clahe.apply(normalized.astype(np.uint8))
        
        return normalized.astype(np.uint8)

    def normalize_all(self, competition_mode=False):
        """Normalize all images with flexible input count"""
        if not self.input_folder:
            messagebox.showerror("Error", "Select input folder or ZIP first!")
            return

        try:
            # Get all image files (sorted naturally)
            image_files = sorted(
                [f for f in os.listdir(self.input_folder) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                key=lambda x: [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', x)]
            )
            
            if not image_files:
                messagebox.showerror("Error", "No images found!")
                return

            # Set reference histogram from first image
            first_img = cv2.imread(os.path.join(self.input_folder, image_files[0]))
            if first_img is None:
                raise ValueError(f"Failed to read image: {image_files[0]}")

            if len(first_img.shape) == 3:
                self.reference_hist = []
                for i in range(3):
                    hist, _ = np.histogram(first_img[:,:,i].flatten(), 256, [0,256])
                    self.reference_hist.append(hist.astype(np.float32))
            else:
                hist, _ = np.histogram(first_img.flatten(), 256, [0,256])
                self.reference_hist = hist.astype(np.float32)
            
            # Competition mode: Use global average
            if competition_mode or self.competition_mode:
                self.target_brightness = self.calculate_global_average(image_files)
                output_folder = self.competition_folder
                # Determine zero-padding length based on total images
                num_length = len(str(len(image_files)))
                prefix = "normalized_image"
                start_idx = 1
            else:
                output_folder = self.output_folder
                prefix = "normalized_"
                start_idx = 0
            
            os.makedirs(output_folder, exist_ok=True)
            self.normalized_images = []
            
            # Process images
            for i, img_file in enumerate(image_files, start_idx):
                img = cv2.imread(os.path.join(self.input_folder, img_file), cv2.IMREAD_ANYCOLOR)
                if img is None:
                    print(f"Skipping {img_file} - could not read image")
                    continue
                    
                normalized = self.enhanced_normalization(img)
                
                if competition_mode or self.competition_mode:
                    output_name = f"{prefix}{str(i).zfill(num_length)}.png"
                else:
                    output_name = f"{prefix}{img_file}"
                
                output_path = os.path.join(output_folder, output_name)
                cv2.imwrite(output_path, normalized)
                self.normalized_images.append(img_file)
                
                orig_mean = self.calculate_perceptual_brightness(img)
                new_mean = self.calculate_perceptual_brightness(normalized)
                print(f"{img_file}: {orig_mean:.1f} → {new_mean:.1f}")

            if competition_mode:
                msg = f"Processed {len(self.normalized_images)} competition images to {output_folder}"
            else:
                msg = f"Normalized {len(self.normalized_images)} images!"
            
            messagebox.showinfo("Success", msg)
            self.show_comparison_gallery()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def add_watermark(self, img, value):
        """Add minimal average intensity watermark to bottom-right corner"""
        if len(img.shape) == 2:  # Convert grayscale to BGR for consistent text color
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        h, w = img.shape[:2]
        text = f"Avg: {value:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        color = (220, 220, 220)  # Light gray
        
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        margin = 5
        pos_x = w - text_w - margin
        pos_y = h - margin
        
        cv2.putText(img, text, (pos_x, pos_y), 
                   font, font_scale, color, thickness, cv2.LINE_AA)
        return img

    def show_comparison_gallery(self):
        """Show scrollable gallery of all normalized images"""
        if not self.normalized_images:
            messagebox.showerror("Error", "No normalized images to display!")
            return

        gallery_win = ctk.CTkToplevel(self)
        gallery_win.title("Normalization Results")
        gallery_win.geometry("720x500")
        
        gallery_win.transient(self)
        gallery_win.grab_set()
        gallery_win.lift()

        canvas = tk.Canvas(gallery_win)
        scrollbar = ctk.CTkScrollbar(gallery_win, orientation="vertical", command=canvas.yview)
        scrollable_frame = ctk.CTkFrame(canvas)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(
            scrollregion=canvas.bbox("all"))
        )
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        for img_file in self.normalized_images:
            frame = ctk.CTkFrame(scrollable_frame)
            frame.pack(fill="x", pady=10, padx=20)
            
            orig_path = os.path.join(self.input_folder, img_file)
            orig_img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
            orig_avg = np.mean(orig_img)
            orig_img = self.add_watermark(orig_img, orig_avg)
            
            if self.competition_mode:
                # Find the corresponding output file
                idx = self.normalized_images.index(img_file) + 1
                num_length = len(str(len(self.normalized_images)))
                norm_filename = f"normalized_image{str(idx).zfill(num_length)}.png"
                norm_path = os.path.join(self.competition_folder, norm_filename)
            else:
                norm_path = os.path.join(self.output_folder, f"normalized_{img_file}")
                
            norm_img = cv2.imread(norm_path, cv2.IMREAD_GRAYSCALE)
            norm_avg = np.mean(norm_img)
            norm_img = self.add_watermark(norm_img, norm_avg)
            
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            norm_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)
            
            orig_img = Image.fromarray(cv2.resize(orig_img, (400, 400)))
            norm_img = Image.fromarray(cv2.resize(norm_img, (400, 400)))
            
            orig_tk = ImageTk.PhotoImage(orig_img)
            norm_tk = ImageTk.PhotoImage(norm_img)
            
            ctk.CTkLabel(frame, text=f"File: {img_file}", 
                        font=("Arial", 12, 'bold')).pack(anchor="w")
            
            img_frame = ctk.CTkFrame(frame, fg_color='transparent')
            img_frame.pack()
            
            lbl_orig = tk.Label(img_frame, image=orig_tk)
            lbl_orig.image = orig_tk
            lbl_orig.grid(row=0, column=0, padx=10)
            
            lbl_norm = tk.Label(img_frame, image=norm_tk)
            lbl_norm.image = norm_tk
            lbl_norm.grid(row=0, column=1, padx=10)
            
            ssim_val = ssim(
                cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE),
                cv2.imread(norm_path, cv2.IMREAD_GRAYSCALE),
                data_range=255
            )
            ctk.CTkLabel(frame, 
                        text=f"Δ Brightness: {norm_avg-orig_avg:+.1f} | SSIM: {ssim_val:.3f}",
                        font=("Arial", 11)).pack(pady=5)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def adjust_single(self):
        """Adjust brightness of a single image"""
        if not self.input_folder:
            messagebox.showerror("Error", "Please select an input folder first!")
            return

        normalized_images = [f for f in os.listdir(self.output_folder) 
                          if f.startswith('normalized_')]
        if not normalized_images:
            messagebox.showerror("Error", "No normalized images found! Run normalization first.")
            return

        adjust_win = ctk.CTkToplevel(self)
        adjust_win.title("Single Image Adjustment")
        adjust_win.geometry("700x600")
        
        adjust_win.transient(self)
        adjust_win.grab_set()
        adjust_win.lift()
        
        ctk.CTkLabel(adjust_win, text="Select Image:", 
                    font=("Arial", 12)).pack(pady=5)
        
        img_selector = ctk.CTkComboBox(adjust_win, 
                                     values=normalized_images,
                                     font=("Arial", 11))
        img_selector.pack(pady=5)
        img_selector.set(normalized_images[0])
        
        ctk.CTkLabel(adjust_win, text="Target Brightness:", 
                    font=("Arial", 12)).pack(pady=5)
        
        self.brightness_var = ctk.StringVar(value=str(self.target_brightness))
        
        def update_slider(val):
            self.brightness_var.set(f"{float(val):.1f}")
            self.update_adjustment_preview(img_selector.get(), float(val))
        
        slider = ctk.CTkSlider(
            adjust_win,
            from_=0,
            to=255,
            number_of_steps=255,
            command=update_slider
        )
        slider.set(self.target_brightness)
        slider.pack(pady=5)
        
        ctk.CTkLabel(adjust_win, 
                    textvariable=self.brightness_var,
                    font=("Arial", 14, 'bold')).pack()
        
        preview_frame = ctk.CTkFrame(adjust_win)
        preview_frame.pack(fill="both", expand=True, pady=15)
        
        self.preview_label = tk.Label(preview_frame)
        self.preview_label.pack()
        
        btn_frame = ctk.CTkFrame(adjust_win, fg_color='transparent')
        btn_frame.pack(pady=10)
        
        ctk.CTkButton(
            btn_frame,
            text="Save Adjusted",
            command=lambda: self.save_adjusted_image(
                img_selector.get(),
                float(self.brightness_var.get()),
                adjust_win
            ),
            width=150,
            height=40,
            font=("Arial", 12)
        ).pack(side="left", padx=20)
        
        self.update_adjustment_preview(img_selector.get(), self.target_brightness)

    def update_adjustment_preview(self, filename, target_brightness):
        """Update the preview image with current adjustment"""
        try:
            img_path = os.path.join(self.output_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            current_mean = np.mean(img)
            factor = target_brightness / (current_mean + 1e-5)
            adjusted = np.clip(img * factor, 0, 255).astype(np.uint8)
            
            adjusted = self.add_watermark(adjusted, target_brightness)
            adjusted = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
            
            adjusted_pil = Image.fromarray(adjusted)
            adjusted_pil.thumbnail((500, 500))
            adjusted_tk = ImageTk.PhotoImage(adjusted_pil)
            
            self.preview_label.config(image=adjusted_tk)
            self.preview_label.image = adjusted_tk
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update preview:\n{str(e)}")

    def save_adjusted_image(self, filename, target_brightness, parent_window):
        """Save the adjusted image"""
        try:
            os.makedirs(self.modified_folder, exist_ok=True)
            
            img_path = os.path.join(self.output_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            current_mean = np.mean(img)
            factor = target_brightness / (current_mean + 1e-5)
            adjusted = np.clip(img * factor, 0, 255).astype(np.uint8)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"adjusted_{filename[10:-4]}_{timestamp}.png"
            output_path = os.path.join(self.modified_folder, new_filename)
            
            cv2.imwrite(output_path, adjusted)
            messagebox.showinfo("Success", f"Saved adjusted image to:\n{output_path}")
            parent_window.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")

if __name__ == "__main__":
    app = ImageNormalizerGUI()
    app.mainloop()