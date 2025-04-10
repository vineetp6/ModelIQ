"""
Advanced Machine Learning Model Comparator & Analyzer

This application allows a user to:
1. Upload a single machine learning model file (supported formats: .pkl, .pt, .pth, .bin,
   .h5, .json, .model, .txt, .onnx) and view an explanation of its parameters.
2. Upload multiple model files, analyze each model to compute statistics such as total parameter
   count and layer count (if applicable), and generate a detailed report.
3. Choose from various chart types (bar chart, pie chart, scatter plot, and line chart) to compare
   the models based on the computed statistics.

Before running this app, ensure that the following packages are installed:
    - PyTorch: pip install torch
    - TensorFlow/Keras: pip install tensorflow
    - ONNX: pip install onnx
    - Matplotlib: pip install matplotlib
"""

import os
import pickle
import json
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt

# Disable oneDNN optimizations to avoid potential numerical differences due to custom ops
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Now you can import TensorFlow safely; the following messages should no longer appear.
import tensorflow as tf

# Your TensorFlow code continues below
# print("TensorFlow version:", tf.__version__)

# Try importing libraries for various model types
try:
    import torch  # For PyTorch models (.pt, .pth, .bin, .model)
except ImportError:
    torch = None

try:
    from tensorflow import keras  # For Keras models (.h5)
except ImportError:
    keras = None

try:
    import onnx  # For ONNX models (.onnx)
except ImportError:
    onnx = None

# ---------------- Loader Functions for Various Model File Types ----------------

def load_pickle_model(filepath):
    """Load a pickle model file (often used by scikit-learn)."""
    with open(filepath, "rb") as file:
        model = pickle.load(file)
    return model

def load_torch_model(filepath):
    """Load a PyTorch model (.pt, .pth, .bin, .model)."""
    if not torch:
        raise ImportError("PyTorch is not installed. Please install torch to load this model.")
    model = torch.load(filepath, map_location="cpu")
    return model

def load_keras_model(filepath):
    """Load a Keras/TensorFlow model saved in HDF5 format (.h5)."""
    if not keras:
        raise ImportError("TensorFlow/Keras is not installed. Please install tensorflow to load this model.")
    model = keras.models.load_model(filepath)
    return model

def load_json_model(filepath):
    """Load a model file in JSON format (may store architecture or configuration)."""
    with open(filepath, "r") as file:
        model = json.load(file)
    return model

def load_onnx_model(filepath):
    """Load an ONNX model (stores the computational graph and initializers)."""
    if not onnx:
        raise ImportError("ONNX is not installed. Please install onnx to load this model.")
    model = onnx.load(filepath)
    return model

def load_text_model(filepath):
    """Load a model stored as plain text (might include configuration details)."""
    with open(filepath, "r") as file:
        content = file.read()
    return content

# ---------------- Functions to Explain and Compute Model Statistics ----------------

def explain_parameters(model_obj, extension):
    """
    Inspects the model object and returns a textual explanation of its parameters.
    Depending on the file type, it lists key information like parameter names,
    types, shapes, and other relevant details.
    """
    explanation = ""
    
    if isinstance(model_obj, dict):
        explanation += "Model Parameters (dictionary format):\n"
        for key, value in model_obj.items():
            shape_info = ""
            if torch and hasattr(value, "shape"):
                shape_info = f" (shape: {tuple(value.shape)})"
            explanation += f"  - {key}{shape_info}: {type(value).__name__}\n"
        explanation += "\nEach key typically represents a weight, bias, or configuration parameter.\n"
    
    elif torch and hasattr(model_obj, "state_dict"):
        explanation += "PyTorch Model detected (full model).\n"
        explanation += "Extracting state_dict parameters:\n"
        state_dict = model_obj.state_dict()
        for key, tensor in state_dict.items():
            explanation += f"  - {key}: type: {type(tensor).__name__}, shape: {tuple(tensor.shape)}\n"
        explanation += "\nThese parameters represent learnable weights used during training and prediction.\n"
    
    elif keras and hasattr(model_obj, "layers"):
        explanation += "Keras Model detected.\n"
        explanation += "Listing layers and weight shapes:\n"
        for layer in model_obj.layers:
            explanation += f"Layer: {layer.name} (class: {layer.__class__.__name__})\n"
            weights = layer.get_weights()
            if weights:
                for idx, weight_array in enumerate(weights):
                    explanation += f"   - Weight {idx}: shape {weight_array.shape}\n"
            else:
                explanation += "   - No weights found for this layer.\n"
        explanation += "\nThese parameters define how the model computes its outputs.\n"
    
    elif onnx and extension.lower() == ".onnx":
        explanation += "ONNX Model detected.\n"
        explanation += "Extracting graph initializers (parameters):\n"
        graph = model_obj.graph
        for init in graph.initializer:
            dims = [d.dim_value for d in init.dims]
            explanation += f"  - {init.name}: shape {dims}, data type {init.data_type}\n"
        explanation += "\nListing model inputs and outputs:\nInputs:\n"
        for inp in graph.input:
            shape = [dim.dim_value if (dim.dim_value > 0) else "None" for dim in inp.type.tensor_type.shape.dim]
            explanation += f"  - {inp.name}: shape {shape}\n"
        explanation += "Outputs:\n"
        for out in graph.output:
            shape = [dim.dim_value if (dim.dim_value > 0) else "None" for dim in out.type.tensor_type.shape.dim]
            explanation += f"  - {out.name}: shape {shape}\n"
        explanation += "\nThese parameters are used for the model's inference computations.\n"
    
    elif extension.lower() == ".json":
        if isinstance(model_obj, dict):
            explanation += "JSON file detected. It contains the following keys:\n"
            for key in model_obj.keys():
                explanation += f"  - {key}\n"
            explanation += "\nThis might represent model architecture or hyperparameter configurations.\n"
        else:
            explanation += "JSON file content:\n" + str(model_obj) + "\n"
    
    elif extension.lower() in [".txt"]:
        explanation += "Text file detected. Content:\n" + model_obj + "\n"
    
    elif isinstance(model_obj, str):
        explanation += "Model loaded as text. Content:\n" + model_obj + "\n"
    
    else:
        explanation += "Model loaded successfully, but its structure could not be automatically interpreted.\n"
    
    explanation += "\n--- Explanation Summary ---\n"
    explanation += "1. **Parameters:** The stored values (e.g., weights, biases) that define the model's behavior.\n"
    explanation += "2. **Operation:** These values are used in the forward pass to compute predictions.\n"
    explanation += "3. **Usage:** They can be tuned during training, inspected for debugging, or used in transfer learning.\n"
    
    return explanation

def compute_parameter_count(model_obj, extension):
    """
    Attempts to compute the total number of parameters for the model.
    Returns an integer count if possible, otherwise None.
    """
    total_params = None

    # For PyTorch: check state_dict and sum numel for each tensor.
    if torch and hasattr(model_obj, "state_dict"):
        total_params = 0
        state_dict = model_obj.state_dict()
        for tensor in state_dict.values():
            if hasattr(tensor, "numel"):
                total_params += tensor.numel()
        return total_params

    # If the model is a dictionary with tensor-like elements.
    if isinstance(model_obj, dict):
        total_params = 0
        for value in model_obj.values():
            if torch and hasattr(value, "numel"):
                total_params += value.numel()
        if total_params != 0:
            return total_params

    # For Keras models, use count_params method.
    if keras and hasattr(model_obj, "count_params"):
        try:
            total_params = model_obj.count_params()
            return total_params
        except Exception:
            pass

    # For ONNX, sum up the product of dimensions for each initializer.
    if onnx and extension.lower() == ".onnx":
        total_params = 0
        for init in model_obj.graph.initializer:
            dims = [d.dim_value for d in init.dims]
            if dims:
                prod = 1
                for d in dims:
                    prod *= d
                total_params += prod
        return total_params

    return None

def compute_layer_count(model_obj, extension):
    """
    Computes the number of layers (or parameter groups) in a model if possible.
    For Keras, it returns the number of layers.
    For PyTorch (if loaded as a full model), it counts the keys in the state_dict.
    For ONNX, it returns the number of initializers.
    """
    layer_count = None

    if torch and hasattr(model_obj, "state_dict"):
        layer_count = len(model_obj.state_dict().keys())
        return layer_count

    if isinstance(model_obj, dict):
        # For dictionary models assumed to be state_dict formats.
        layer_count = len(model_obj.keys())
        if layer_count > 0:
            return layer_count

    if keras and hasattr(model_obj, "layers"):
        return len(model_obj.layers)

    if onnx and extension.lower() == ".onnx":
        return len(model_obj.graph.initializer)

    return None

# ---------------- GUI Functions ----------------

def upload_and_explain():
    """Handles single model upload and displays an explanation of its parameters."""
    filepath = filedialog.askopenfilename(
        title="Select a Machine Learning Model File",
        filetypes=[("Supported Files", "*.pkl *.pt *.pth *.bin *.h5 *.json *.model *.txt *.onnx"), ("All Files", "*.*")]
    )
    if not filepath:
        return

    _, ext = os.path.splitext(filepath)
    ext = ext.lower()

    try:
        if ext == ".pkl":
            model_obj = load_pickle_model(filepath)
        elif ext in [".pt", ".pth", ".bin", ".model"]:
            model_obj = load_torch_model(filepath)
        elif ext == ".h5":
            model_obj = load_keras_model(filepath)
        elif ext == ".json":
            model_obj = load_json_model(filepath)
        elif ext == ".onnx":
            model_obj = load_onnx_model(filepath)
        elif ext == ".txt":
            model_obj = load_text_model(filepath)
        else:
            messagebox.showerror("Error", f"Unsupported file extension: {ext}")
            return

        explanation = explain_parameters(model_obj, ext)
        output_text.config(state=tk.NORMAL)
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, explanation)
        output_text.config(state=tk.DISABLED)

    except Exception as e:
        messagebox.showerror("Error", f"Error processing the file:\n{str(e)}")

def compare_and_analyze_models():
    """
    Allows the user to select multiple model files, computes various metrics for each model,
    generates a detailed text report, and enables selection from different chart types for visualization.
    """
    filepaths = filedialog.askopenfilenames(
        title="Select Multiple Machine Learning Model Files",
        filetypes=[("Supported Files", "*.pkl *.pt *.pth *.bin *.h5 *.json *.model *.txt *.onnx"), ("All Files", "*.*")]
    )
    if not filepaths:
        return

    report_lines = []
    model_names = []
    param_counts = []
    layer_counts = []
    avg_params_per_layer = []

    errors = []

    # Process each selected file.
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()

        try:
            if ext == ".pkl":
                model_obj = load_pickle_model(filepath)
            elif ext in [".pt", ".pth", ".bin", ".model"]:
                model_obj = load_torch_model(filepath)
            elif ext == ".h5":
                model_obj = load_keras_model(filepath)
            elif ext == ".json":
                model_obj = load_json_model(filepath)
            elif ext == ".onnx":
                model_obj = load_onnx_model(filepath)
            elif ext == ".txt":
                model_obj = load_text_model(filepath)
            else:
                errors.append(f"{filename}: Unsupported extension.")
                continue

            # Compute statistics
            p_count = compute_parameter_count(model_obj, ext)
            l_count = compute_layer_count(model_obj, ext)
            avg_param = None
            if p_count is not None and l_count and l_count > 0:
                avg_param = p_count / l_count

            model_names.append(filename)
            param_counts.append(p_count if p_count is not None else 0)
            layer_counts.append(l_count if l_count is not None else 0)
            avg_params_per_layer.append(avg_param if avg_param is not None else 0)

            # Build detailed report for this model.
            report_lines.append(f"Model: {filename}")
            report_lines.append(f"  - Total Parameters: {p_count if p_count is not None else 'N/A'}")
            report_lines.append(f"  - Layer Count: {l_count if l_count is not None else 'N/A'}")
            report_lines.append(f"  - Average Parameters per Layer: {avg_param if avg_param is not None else 'N/A'}")
            report_lines.append("-" * 60)

        except Exception as e:
            errors.append(f"{filename}: Error -> {str(e)}")
            continue

    if not model_names:
        messagebox.showerror("Error", "No models with computable statistics were selected.\n" + "\n".join(errors))
        return

    # Display detailed report in the text area.
    detailed_report = "\n".join(report_lines)
    output_text.config(state=tk.NORMAL)
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, detailed_report)
    output_text.config(state=tk.DISABLED)

    # Create a popup window for chart selection.
    chart_window = tk.Toplevel(root)
    chart_window.title("Select Chart Type for Comparison")
    chart_window.geometry("300x150")

    chart_label = tk.Label(chart_window, text="Choose a chart type:")
    chart_label.pack(pady=10)

    # Options for chart types.
    chart_options = [
        "Bar Chart (Parameter Count)",
        "Pie Chart (Parameter Distribution)",
        "Scatter Plot (Parameters vs. Layers)",
        "Line Chart (Parameter Trend)"
    ]
    chart_var = tk.StringVar(chart_window)
    chart_var.set(chart_options[0])  # Default option

    chart_menu = tk.OptionMenu(chart_window, chart_var, *chart_options)
    chart_menu.pack(pady=5)

    def show_chart():
        chart_type = chart_var.get()
        plt.figure(figsize=(10, 6))
        if chart_type == "Bar Chart (Parameter Count)":
            positions = range(len(model_names))
            plt.bar(positions, param_counts, color='skyblue')
            plt.xticks(positions, model_names, rotation=45, ha="right")
            plt.ylabel("Total Parameters")
            plt.title("Model Comparison: Total Parameter Count")
        elif chart_type == "Pie Chart (Parameter Distribution)":
            plt.pie(param_counts, labels=model_names, autopct='%1.1f%%', startangle=140)
            plt.title("Model Comparison: Parameter Distribution")
        elif chart_type == "Scatter Plot (Parameters vs. Layers)":
            plt.scatter(layer_counts, param_counts, color='green')
            for i, name in enumerate(model_names):
                plt.annotate(name, (layer_counts[i], param_counts[i]))
            plt.xlabel("Layer Count")
            plt.ylabel("Total Parameters")
            plt.title("Scatter Plot: Total Parameters vs. Layer Count")
        elif chart_type == "Line Chart (Parameter Trend)":
            positions = range(len(model_names))
            plt.plot(positions, param_counts, marker='o', linestyle='-', color='orange')
            plt.xticks(positions, model_names, rotation=45, ha="right")
            plt.ylabel("Total Parameters")
            plt.title("Line Chart: Parameter Trend Across Models")
        plt.tight_layout()
        plt.show()

    show_button = tk.Button(chart_window, text="Show Chart", command=show_chart)
    show_button.pack(pady=10)

    # Optionally, display errors from processing (if any)
    if errors:
        err_msg = "Some models could not be processed:\n" + "\n".join(errors)
        messagebox.showwarning("Processing Warnings", err_msg)

# ---------------- Main GUI Setup ----------------

root = tk.Tk()
root.title("Advanced Model Comparator & Analyzer")
root.geometry("950x700")

# Frame for buttons.
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Button to upload a single model for explanation.
upload_button = tk.Button(button_frame, text="Upload & Explain Model", command=upload_and_explain, font=("Arial", 12))
upload_button.grid(row=0, column=0, padx=5, pady=5)

# Button to compare and analyze multiple models.
compare_button = tk.Button(button_frame, text="Compare & Analyze Models", command=compare_and_analyze_models, font=("Arial", 12))
compare_button.grid(row=0, column=1, padx=5, pady=5)

# Scrolled text area for displaying explanations or detailed reports.
output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Courier New", 10))
output_text.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
output_text.config(state=tk.DISABLED)

# Start the Tkinter main loop.
root.mainloop()
