#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EEG Emotion Recognition Models GUI Application
This application provides a graphical interface to:
- Select between TensorFlow and PyTorch implementations
- Choose modality (EEG only or Multimodal, or both)
- Choose feature extraction method (FFT or Welch, or both)
- Select frequency bands (delta, theta, alpha, beta, gamma, overall)
- Select models to train
- Set preferences (dark/light mode, input/output file path)
- Monitor script execution in real-time
- Start, pause and cancel script execution
"""

import sys
import os
import subprocess
import threading
import json
import psutil  # Added for process management
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QRadioButton, QButtonGroup, QPushButton, QLabel, QLineEdit,
    QTextEdit, QFileDialog, QGroupBox, QProgressBar, QMessageBox, QStatusBar,
    QCheckBox, QTabWidget, QScrollArea, QGridLayout
)
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, Qt, QEvent
from PyQt5.QtGui import QFont, QTextCursor, QPalette, QColor


# Stylesheet definitions for dark and light themes
DARK_STYLE = """
QWidget {
    background-color: #2D2D30;
    color: #FFFFFF;
}
QTabWidget::pane {
    border: 1px solid #3E3E40;
    background-color: #2D2D30;
}
QTabBar::tab {
    background-color: #252526;
    color: #FFFFFF;
    padding: 8px 16px;
    border: 1px solid #3E3E40;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background-color: #3E3E42;
}
QGroupBox {
    border: 1px solid #3E3E40;
    margin-top: 1ex;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    padding: 0px 5px;
}
QPushButton {
    background-color: #0E639C;
    color: white;
    border: none;
    padding: 5px 10px;
    border-radius: 3px;
}
QPushButton:hover {
    background-color: #1177BB;
}
QPushButton:pressed {
    background-color: #0D5A8C;
}
QPushButton:disabled {
    background-color: #3E3E40;
    color: #888888;
}
QLineEdit {
    background-color: #1E1E1E;
    color: white;
    border: 1px solid #3E3E40;
    padding: 5px;
    border-radius: 3px;
}
QTextEdit {
    background-color: #1E1E1E;
    color: white;
    border: 1px solid #3E3E40;
}
QCheckBox {
    color: white;
}
QRadioButton {
    color: white;
}
QProgressBar {
    border: 1px solid #3E3E40;
    border-radius: 3px;
    text-align: center;
    background-color: #1E1E1E;
}
QProgressBar::chunk {
    background-color: #0E639C;
}
"""

LIGHT_STYLE = """
QWidget {
    background-color: #F0F0F0;
    color: #000000;
}
QTabWidget::pane {
    border: 1px solid #C0C0C0;
    background-color: #F0F0F0;
}
QTabBar::tab {
    background-color: #E0E0E0;
    color: #000000;
    padding: 8px 16px;
    border: 1px solid #C0C0C0;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background-color: #FFFFFF;
}
QGroupBox {
    border: 1px solid #C0C0C0;
    margin-top: 1ex;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    padding: 0px 5px;
}
QPushButton {
    background-color: #0078D7;
    color: white;
    border: none;
    padding: 5px 10px;
    border-radius: 3px;
}
QPushButton:hover {
    background-color: #1089E7;
}
QPushButton:pressed {
    background-color: #006CC7;
}
QPushButton:disabled {
    background-color: #C0C0C0;
    color: #888888;
}
QLineEdit {
    background-color: white;
    color: black;
    border: 1px solid #C0C0C0;
    padding: 5px;
    border-radius: 3px;
}
QTextEdit {
    background-color: white;
    color: black;
    border: 1px solid #C0C0C0;
}
QCheckBox {
    color: black;
}
QRadioButton {
    color: black;
}
QProgressBar {
    border: 1px solid #C0C0C0;
    border-radius: 3px;
    text-align: center;
    background-color: white;
}
QProgressBar::chunk {
    background-color: #0078D7;
}
"""


class StreamRedirector(QObject):
    """Redirects Python's stdout/stderr to a Qt signal."""
    text_written = pyqtSignal(str)
    
    def __init__(self, original_stream):
        super().__init__()
        self.original_stream = original_stream
        
    def write(self, text):
        if text.strip():  # Only emit non-empty strings
            self.text_written.emit(text)
        self.original_stream.write(text)
        
    def flush(self):
        self.original_stream.flush()


class Executor(QMainWindow):
    """Main window"""
    
    def __init__(self):
        super().__init__()
        self.process = None
        self.execution_thread = None
        self.running = False
        self.paused = False  # Flag to track pause state
        self.pause_event = threading.Event()  # Event to signal pause/resume
        self.pause_event.set()  # Not paused by default
        self.preferences_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "eeg_emorec_preferences.json"
        )
        self.dark_mode = True  # Default to dark mode
        self.selected_models = ["SVM", "RF", "Decision Tree", "CNN", "FCNN", 
                                "FCNN+Attention", "Domain-Adversarial Fuzzy", "GraphCNN"]
        self.dreamer_mat_path = "DREAMER.mat"
        self.output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "results", 
            "results.txt"
        )
        self.load_preferences()
        self.initUI()
        self.apply_theme()
        
    def initUI(self):
        self.setWindowTitle('EEG Emotion Recognition Models v2.0') 
        self.setGeometry(100, 100, 900, 700)
        
        # Main layout with tab widget
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)
        
        self.tab_widget = QTabWidget()
        self.main_tab = QWidget()
        self.preferences_tab = QWidget()
        
        self.tab_widget.addTab(self.main_tab, "Main")
        self.tab_widget.addTab(self.preferences_tab, "Preferences")
        
        # Set up the main tab
        self.setup_main_tab()
        
        # Set up the preferences tab
        self.setup_preferences_tab()
        
        # Add tab widget to main layout
        self.main_layout.addWidget(self.tab_widget)
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('Ready')
        
        # Set up stdout/stderr redirection
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        self.stdout_redirector = StreamRedirector(self.original_stdout)
        self.stderr_redirector = StreamRedirector(self.original_stderr)
        
        self.stdout_redirector.text_written.connect(self.update_terminal)
        self.stderr_redirector.text_written.connect(self.update_terminal)
        
        # Show the main window
        self.show()
    
    def setup_main_tab(self):
        """Set up the main tab with all controls."""
        main_layout = QVBoxLayout(self.main_tab)
        
        # Framework selection group
        framework_group = QGroupBox("Framework Selection")
        framework_layout = QHBoxLayout()
        
        self.tf_radio = QRadioButton("TensorFlow")
        self.pytorch_radio = QRadioButton("PyTorch")
        self.tf_radio.setChecked(True)  # Default to TensorFlow
        
        self.framework_group = QButtonGroup()
        self.framework_group.addButton(self.tf_radio, 1)
        self.framework_group.addButton(self.pytorch_radio, 2)
        
        framework_layout.addWidget(self.tf_radio)
        framework_layout.addWidget(self.pytorch_radio)
        framework_group.setLayout(framework_layout)
        
        # Modality selection group
        modality_group = QGroupBox("Modality Selection")
        modality_layout = QHBoxLayout()
        
        self.eeg_only_check = QCheckBox("EEG only")
        self.multimodal_check = QCheckBox("Multimodal")
        self.eeg_only_check.setChecked(True)  # Default to EEG only
        
        modality_layout.addWidget(self.eeg_only_check)
        modality_layout.addWidget(self.multimodal_check)
        modality_group.setLayout(modality_layout)
        
        # Feature extraction method selection group
        feature_method_group = QGroupBox("Feature Extraction Method")
        feature_method_layout = QHBoxLayout()
        
        self.fft_check = QCheckBox("FFT")
        self.welch_check = QCheckBox("Welch")
        self.fft_check.setChecked(True)  # Default to FFT
        
        feature_method_layout.addWidget(self.fft_check)
        feature_method_layout.addWidget(self.welch_check)
        feature_method_group.setLayout(feature_method_layout)
        
        # Frequency band selection group - now displayed horizontally
        band_group = QGroupBox("Frequency Band Selection")
        band_layout = QHBoxLayout()
        
        self.band_checkboxes = {}
        bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma", "Overall"]
        
        for band in bands:
            checkbox = QCheckBox(band)
            if band == "Overall":
                checkbox.setChecked(True)  # Default to overall
            self.band_checkboxes[band] = checkbox
            band_layout.addWidget(checkbox)
        
        band_group.setLayout(band_layout)
        
        # Model selection group - moved from preferences to main tab
        model_group = QGroupBox("Model Selection")
        model_layout = QGridLayout()
        
        self.model_checkboxes = {}
        models = ["SVM", "RF", "Decision Tree", "CNN", "FCNN", 
                  "FCNN+Attention", "Domain-Adversarial Fuzzy", "GraphCNN"]
        
        # Grid layout with 4 models per row
        for i, model in enumerate(models):
            checkbox = QCheckBox(model)
            checkbox.setChecked(model in self.selected_models)
            checkbox.toggled.connect(self.update_model_selection)
            self.model_checkboxes[model] = checkbox
            row, col = divmod(i, 4)
            model_layout.addWidget(checkbox, row, col)
        
        # Select/Deselect All buttons
        model_button_layout = QHBoxLayout()
        select_all_button = QPushButton("Select All")
        deselect_all_button = QPushButton("Deselect All")
        
        select_all_button.clicked.connect(self.select_all_models)
        deselect_all_button.clicked.connect(self.deselect_all_models)
        
        model_button_layout.addWidget(select_all_button)
        model_button_layout.addWidget(deselect_all_button)
        
        # Add the buttons to the grid at the bottom
        model_layout.addLayout(model_button_layout, (len(models) + 3) // 4, 0, 1, 4)
        model_group.setLayout(model_layout)
        
        # Terminal/console output
        terminal_group = QGroupBox("Console Output")
        terminal_layout = QVBoxLayout()
        
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setFont(QFont("Courier", 10))
        
        terminal_layout.addWidget(self.terminal)
        terminal_group.setLayout(terminal_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_script)
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setEnabled(False)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_script)
        self.cancel_button.setEnabled(False)
        
        exit_button = QPushButton("Exit")
        exit_button.clicked.connect(self.close)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch(1)
        button_layout.addWidget(exit_button)
        
        # Add groups to main layout
        main_layout.addWidget(framework_group)
        main_layout.addWidget(modality_group)
        main_layout.addWidget(feature_method_group)
        main_layout.addWidget(band_group)
        main_layout.addWidget(model_group)
        main_layout.addWidget(terminal_group)
        main_layout.addWidget(self.progress_bar)
        main_layout.addLayout(button_layout)
    
    def setup_preferences_tab(self):
        """Set up the preferences tab with theme and output file selection."""
        preferences_layout = QVBoxLayout(self.preferences_tab)
        
        # Theme selection group
        theme_group = QGroupBox("Theme Selection")
        theme_layout = QHBoxLayout()
        
        self.dark_radio = QRadioButton("Dark Mode")
        self.light_radio = QRadioButton("Light Mode")
        
        # Set the selected theme based on preferences
        if self.dark_mode:
            self.dark_radio.setChecked(True)
        else:
            self.light_radio.setChecked(True)
            
        # Connect theme radio buttons to change theme
        self.dark_radio.toggled.connect(self.theme_changed)
        self.light_radio.toggled.connect(self.theme_changed)
        
        theme_layout.addWidget(self.dark_radio)
        theme_layout.addWidget(self.light_radio)
        theme_group.setLayout(theme_layout)
        
        # DREAMER.mat file selection
        dreamer_group = QGroupBox("DREAMER.mat File")
        dreamer_layout = QHBoxLayout()
        
        self.dreamer_path_field = QLineEdit()
        self.dreamer_path_field.setText(self.dreamer_mat_path)
        
        dreamer_browse_button = QPushButton("Browse...")
        dreamer_browse_button.clicked.connect(self.browse_dreamer_file)
        
        dreamer_layout.addWidget(self.dreamer_path_field)
        dreamer_layout.addWidget(dreamer_browse_button)
        dreamer_group.setLayout(dreamer_layout)
        
        # Output file selection - moved from main tab to preferences tab
        output_group = QGroupBox("Output File")
        output_layout = QHBoxLayout()
        
        self.output_path_field = QLineEdit()
        self.output_path_field.setText(self.output_path)
        
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_file)
        
        output_layout.addWidget(self.output_path_field)
        output_layout.addWidget(browse_button)
        output_group.setLayout(output_layout)
        
        # Save preferences button
        save_button = QPushButton("Save Preferences")
        save_button.clicked.connect(self.save_preferences)
        
        # Add groups to preferences layout
        preferences_layout.addWidget(theme_group)
        preferences_layout.addWidget(dreamer_group)
        preferences_layout.addWidget(output_group)
        preferences_layout.addWidget(save_button)
        preferences_layout.addStretch(1)  # Add stretch to push everything to the top
    
    def theme_changed(self):
        """Handle theme change."""
        self.dark_mode = self.dark_radio.isChecked()
        self.apply_theme()
        
    def apply_theme(self):
        """Apply the selected theme to the application."""
        if self.dark_mode:
            self.setStyleSheet(DARK_STYLE)
            # Set terminal colors for dark mode
            self.terminal.setStyleSheet("background-color: #1E1E1E; color: white;")
        else:
            self.setStyleSheet(LIGHT_STYLE)
            # Set terminal colors for light mode
            self.terminal.setStyleSheet("background-color: white; color: black;")
    
    def update_model_selection(self):
        """Update the selected models list based on checkboxes."""
        self.selected_models = [model for model, checkbox in self.model_checkboxes.items() 
                               if checkbox.isChecked()]
    
    def select_all_models(self):
        """Select all models."""
        for checkbox in self.model_checkboxes.values():
            checkbox.setChecked(True)
    
    def deselect_all_models(self):
        """Deselect all models."""
        for checkbox in self.model_checkboxes.values():
            checkbox.setChecked(False)
    
    def save_preferences(self):
        """Save preferences to a JSON file."""
        try:
            # Update paths from the fields
            self.output_path = self.output_path_field.text()
            self.dreamer_mat_path = self.dreamer_path_field.text()
            
            preferences = {
                "dark_mode": self.dark_mode,
                "selected_models": self.selected_models,
                "output_path": self.output_path,
                "dreamer_mat_path": self.dreamer_mat_path
            }
            
            with open(self.preferences_file, 'w') as f:
                json.dump(preferences, f)
                
            self.statusBar.showMessage("Preferences saved successfully")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not save preferences: {str(e)}")
    
    def load_preferences(self):
        """Load preferences from JSON file if it exists."""
        try:
            if os.path.exists(self.preferences_file):
                with open(self.preferences_file, 'r') as f:
                    preferences = json.load(f)
                
                self.dark_mode = preferences.get("dark_mode", True)
                self.selected_models = preferences.get("selected_models", 
                                                      ["SVM", "RF", "Decision Tree", "CNN", "FCNN", 
                                                       "FCNN+Attention", "Domain-Adversarial Fuzzy", "GraphCNN"])
                self.output_path = preferences.get("output_path", self.output_path)
                self.dreamer_mat_path = preferences.get("dreamer_mat_path", "DREAMER.mat")
        except Exception as e:
            print(f"Error loading preferences: {str(e)}")
    
    def browse_file(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Output File", self.output_path_field.text(),
            "Text Files (*.txt);;All Files (*)")
        if file_path:
            self.output_path_field.setText(file_path)
            self.output_path = file_path
            self.statusBar.showMessage(f'Output file set to {file_path}')
    
    def browse_dreamer_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select DREAMER.mat File", self.dreamer_path_field.text(),
            "MAT Files (*.mat);;All Files (*)")
        if file_path:
            self.dreamer_path_field.setText(file_path)
            self.dreamer_mat_path = file_path
            self.statusBar.showMessage(f'DREAMER.mat file set to {file_path}')
    
    @pyqtSlot(str)
    def update_terminal(self, text):
        """Update the terminal with new output."""
        self.terminal.moveCursor(QTextCursor.End)
        self.terminal.insertPlainText(text)
        self.terminal.moveCursor(QTextCursor.End)
    
    def get_selected_modalities(self):
        """Get the list of selected modalities."""
        modalities = []
        if self.eeg_only_check.isChecked():
            modalities.append("eeg_only")
        if self.multimodal_check.isChecked():
            modalities.append("multimodal")
        return modalities
    
    def get_selected_feature_methods(self):
        """Get the list of selected feature extraction methods."""
        methods = []
        if self.fft_check.isChecked():
            methods.append("fft")
        if self.welch_check.isChecked():
            methods.append("welch")
        return methods
    
    def get_selected_bands(self):
        """Get the list of selected frequency bands."""
        return [band for band, checkbox in self.band_checkboxes.items() if checkbox.isChecked()]
    
    def start_script(self):
        """Start execution of the selected script."""
        if self.running:
            return
        
        # Make sure output path is updated from preferences tab
        self.output_path = self.output_path_field.text()
        
        # Get selected options
        framework = "tensorflow" if self.tf_radio.isChecked() else "pytorch"
        modalities = self.get_selected_modalities()
        feature_methods = self.get_selected_feature_methods()
        selected_bands = self.get_selected_bands()
        output_file = self.output_path
        
        # Validation checks
        if not modalities:
            QMessageBox.warning(self, "Warning", "Please select at least one modality.")
            return
        
        if not feature_methods:
            QMessageBox.warning(self, "Warning", "Please select at least one feature extraction method.")
            return
        
        if len(selected_bands) == 0:
            QMessageBox.warning(self, "Warning", "Please select at least one frequency band.")
            return
        
        if not self.selected_models:
            QMessageBox.warning(self, "Warning", "Please select at least one model to train.")
            return
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Error", f"Failed to create output directory: {str(e)}")
            return
        
        # Determine script to run
        script_mapping = {
            "tensorflow": "AIOtensorflow.py", 
            "pytorch": "AIOpytorch.py"
        }
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_mapping[framework])
        
        if not os.path.exists(script_path):
            QMessageBox.critical(self, "Error", f"Script not found: {script_path}")
            return
        
        # Clear terminal and update UI
        self.terminal.clear()
        self.terminal.append(f"Starting {framework.capitalize()} implementation...\n")
        self.terminal.append(f"Modalities: {', '.join(m.replace('_', ' ').title() for m in modalities)}\n")
        self.terminal.append(f"Feature methods: {', '.join(m.upper() for m in feature_methods)}\n")
        self.terminal.append(f"Selected bands: {', '.join(selected_bands)}\n")
        self.terminal.append(f"Selected models: {', '.join(self.selected_models)}\n")
        self.terminal.append(f"Output will be saved to: {output_file}\n\n")
        
        # Update UI state
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.statusBar.showMessage(f'Running {framework.capitalize()} implementation...')
        self.running = True
        
        # Create stop event for the process monitor
        self.stop_monitor = threading.Event()
        
        # Redirect stdout/stderr
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stderr_redirector
        
        # Start execution in a separate thread for each combination
        self.execution_threads = []
        
        for modality in modalities:
            for feat_method in feature_methods:
                # Create a unique output file name based on the combination
                base_filename, extension = os.path.splitext(output_file)
                combination_output = f"{base_filename}_{modality}_{feat_method}{extension}"
                
                thread = threading.Thread(
                    target=self.run_script, 
                    args=(script_path, combination_output, modality, feat_method, selected_bands)
                )
                thread.daemon = True
                self.execution_threads.append(thread)
                thread.start()
                
                # Log the start of this specific combination
                self.terminal.append(f"Starting combination: {modality} + {feat_method} -> {combination_output}\n")
                
        # Start a monitor thread to detect process termination
        self.monitor_thread = threading.Thread(target=self.monitor_process)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def run_script(self, script_path, output_file, modality, feat_method, selected_bands):
        """Execute the script in a subprocess."""
        try:
            # Prepare the command with additional arguments
            cmd = [
                sys.executable, 
                script_path,
                "--modality", modality,
                "--feat_method", feat_method,
                "--bands", ",".join(selected_bands),
                "--output", output_file,
                "--models", ",".join(self.selected_models),
                "--dreamer_mat", self.dreamer_mat_path
            ]
            
            # Execute the script
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                cwd=os.path.dirname(script_path)
            )
            
            # Flag to track if we're canceled
            is_canceled = False
            
            # Capture and redirect output
            with open(output_file, 'w', encoding='utf-8') as f:
                # Store a local reference to the process to avoid NoneType issues
                process_ref = self.process
                if process_ref is None:  # Process was already canceled
                    is_canceled = True
                    return
                    
                try:
                    for line in process_ref.stdout:
                        # Check if process was canceled or set to None
                        if self.process is None:
                            is_canceled = True
                            break
                            
                        # Check for pause event - if cleared, wait until set
                        self.pause_event.wait()
                        
                        if line:
                            print(line, end='')  # This will go through our redirector
                            f.write(line)
                            f.flush()
                except (ValueError, IOError, BrokenPipeError) as e:
                    # These exceptions can occur if the process is terminated
                    is_canceled = True
                    self.terminal.append(f"\n*** Output processing interrupted: {str(e)} ***\n")
                
                # Check if process is still running - it may have exited with error
                if process_ref.poll() is not None:
                    return_code = process_ref.returncode
                    if return_code != 0:
                        self.terminal.append(f"\n*** Process exited with error code {return_code} ***\n")
                        self.reset_execution_state()
            
            # Only wait on the process if it wasn't canceled and still exists
            if not is_canceled and self.process is not None:
                try:
                    return_code = process_ref.wait()
                    if return_code == 0:
                        self.update_status(f"Script completed successfully: {os.path.basename(output_file)}")
                    else:
                        self.update_status(f"Script exited with code {return_code}: {os.path.basename(output_file)}")
                except (ValueError, AttributeError) as e:
                    # Process was terminated or is None
                    self.terminal.append(f"\n*** Process was terminated: {str(e)} ***\n")
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            self.terminal.append(f"\n*** Exception occurred: {str(e)} ***\n")
            # Reset execution state on exception
            self.reset_execution_state()
        finally:
            # Check if all threads are completed
            active_threads = [t for t in self.execution_threads if t.is_alive()]
            if not active_threads:
                # Reset the state if all threads are done
                self.reset_execution_state()
                self.update_ui_after_execution()
    
    def update_status(self, message):
        """Update status bar with message (thread-safe)."""
        QApplication.postEvent(self, 
                               StatusUpdateEvent(message))
    
    def update_ui_after_execution(self):
        """Reset UI elements after script execution (thread-safe)."""
        QApplication.postEvent(self, UIUpdateEvent())
    
    def cancel_script(self):
        """Cancel the running script and properly clean up resources."""
        if not self.running:
            return
            
        try:
            if self.process:
                # Get the process ID
                pid = self.process.pid
                
                # Kill the process and any child processes
                parent = psutil.Process(pid)
                for child in parent.children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                
                # Kill the parent process
                try:
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass
                    
                self.process = None
            
            # Clear pause event to allow any waiting threads to proceed and terminate
            self.pause_event.set()
            
            # Wait for a short time for threads to clean up
            self.terminal.append("\n\n*** Script execution canceled by user ***\n")
            self.update_status("Execution canceled")
            
            # Reset state immediately
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
            self.progress_bar.setVisible(False)
            self.running = False
            self.paused = False
            self.pause_button.setText("Pause")
            
            # Reset stdout/stderr
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            
        except Exception as e:
            self.terminal.append(f"\n\n*** Error during cancellation: {str(e)} ***\n")

        # Force garbage collection to free memory
        import gc
        gc.collect()
        
        self.statusBar.showMessage("Ready")    
        
    def handle_ui_update(self):
        """Handle UI updates after script execution."""
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.running = False
        
    def event(self, event):
        if hasattr(event, 'type') and event.type() == StatusUpdateEvent.EVENT_TYPE:
            self.statusBar.showMessage(event.message)
            return True
        elif hasattr(event, 'type') and event.type() == UIUpdateEvent.EVENT_TYPE:
            self.handle_ui_update()
            return True
        return super().event(event)
        
    def customEvent(self, event):
        if event.type() == UIUpdateEvent.EVENT_TYPE:
            self.handle_ui_update()
            return True
        if event.type() == StatusUpdateEvent.EVENT_TYPE:
            self.statusBar.showMessage(event.message)
            return True
        return super().customEvent(event)
    
    def closeEvent(self, event):
        if self.running:
            reply = QMessageBox.question(
                self, 'Confirm Exit', 
                'A script is still running. Terminate and exit?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.cancel_script()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def toggle_pause(self):
        """Toggle between pause and resume states for the running process."""
        if not self.running or not self.process:
            return
            
        self.paused = not self.paused
        
        if self.paused:
            # Pause execution - actually suspend the process
            try:
                # Get the process and all children
                parent = psutil.Process(self.process.pid)
                processes = [parent] + parent.children(recursive=True)
                
                # Suspend all processes
                for proc in processes:
                    try:
                        proc.suspend()
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        self.terminal.append(f"\n*** Warning: Could not suspend process {proc.pid}: {str(e)} ***\n")
                
                self.pause_event.clear()  # Also block the output reading thread
                self.pause_button.setText("Resume")
                self.statusBar.showMessage("Execution paused")
                self.terminal.append("\n*** Execution paused. Process suspended. Click Resume to continue ***\n")
            except Exception as e:
                self.terminal.append(f"\n*** Error pausing execution: {str(e)} ***\n")
        else:
            # Resume execution - resume the suspended process
            try:
                # Get the process and all children
                parent = psutil.Process(self.process.pid)
                processes = [parent] + parent.children(recursive=True)
                
                # Resume all processes
                for proc in processes:
                    try:
                        proc.resume()
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        self.terminal.append(f"\n*** Warning: Could not resume process {proc.pid}: {str(e)} ***\n")
                
                self.pause_event.set()  # Unblock the output reading thread
                self.pause_button.setText("Pause")
                self.statusBar.showMessage("Execution resumed")
                self.terminal.append("\n*** Resuming execution. Process continuing... ***\n")
            except Exception as e:
                self.terminal.append(f"\n*** Error resuming execution: {str(e)} ***\n")

    def reset_execution_state(self):
        """Reset the execution state after script completion or error."""
        # Set running flag to false
        self.running = False
        
        # Reset process reference
        self.process = None
        
        # Reset pause state
        self.paused = False
        self.pause_event.set()
        self.pause_button.setText("Pause")
        
        # Update UI elements
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        # Reset stdout/stderr redirections
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        # Force garbage collection
        import gc
        gc.collect()

    def monitor_process(self):
        """Monitor the process and update UI when it terminates."""
        while not self.stop_monitor.is_set():
            if self.process and self.process.poll() is not None:
                self.update_status("Process terminated")
                self.reset_execution_state()
                self.update_ui_after_execution()
                break
            self.stop_monitor.wait(1)


class StatusUpdateEvent(QEvent):
    """Custom event for updating status bar from another thread."""
    EVENT_TYPE = QEvent.Type(QEvent.registerEventType())
    
    def __init__(self, message):
        super().__init__(StatusUpdateEvent.EVENT_TYPE)
        self.message = message


class UIUpdateEvent(QEvent):
    """Custom event for updating UI elements from another thread."""
    EVENT_TYPE = QEvent.Type(QEvent.registerEventType())
    
    def __init__(self):
        super().__init__(UIUpdateEvent.EVENT_TYPE)


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    ex = Executor()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
