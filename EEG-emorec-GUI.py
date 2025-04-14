#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EEG Emotion Recognition Models GUI Application
This application provides a graphical interface to:
- Select between TensorFlow and PyTorch implementations
- Choose modality (EEG only or Multimodal, or both)
- Choose feature extraction method (FFT or Welch, or both)
- Select frequency bands (delta, theta, alpha, beta, gamma, overall)
- Specify an output file path for results
- Monitor script execution in real-time
- Start and cancel script execution
"""

import sys
import os
import subprocess
import threading
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QRadioButton, QButtonGroup, QPushButton, QLabel, QLineEdit,
    QTextEdit, QFileDialog, QGroupBox, QProgressBar, QMessageBox, QStatusBar,
    QCheckBox
)
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, Qt, QEvent
from PyQt5.QtGui import QFont, QTextCursor


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
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('EEG Emotion Recognition Models v1.0') 
        self.setGeometry(100, 100, 800, 600)
        
        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
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
        
        # Modality selection group - changed to checkboxes to allow multi-selection
        modality_group = QGroupBox("Modality Selection")
        modality_layout = QHBoxLayout()
        
        self.eeg_only_check = QCheckBox("EEG only")
        self.multimodal_check = QCheckBox("Multimodal")
        self.eeg_only_check.setChecked(True)  # Default to EEG only
        
        modality_layout.addWidget(self.eeg_only_check)
        modality_layout.addWidget(self.multimodal_check)
        modality_group.setLayout(modality_layout)
        
        # Feature extraction method selection group - changed to checkboxes
        feature_method_group = QGroupBox("Feature Extraction Method")
        feature_method_layout = QHBoxLayout()
        
        self.fft_check = QCheckBox("FFT")
        self.welch_check = QCheckBox("Welch")
        self.fft_check.setChecked(True)  # Default to FFT
        
        feature_method_layout.addWidget(self.fft_check)
        feature_method_layout.addWidget(self.welch_check)
        feature_method_group.setLayout(feature_method_layout)
        
        # Frequency band selection group
        band_group = QGroupBox("Frequency Band Selection")
        band_layout = QVBoxLayout()
        
        self.band_checkboxes = {}
        bands = ["delta", "theta", "alpha", "beta", "gamma", "overall"]
        
        for band in bands:
            checkbox = QCheckBox(band)
            if band == "overall":
                checkbox.setChecked(True)  # Default to overall
            self.band_checkboxes[band] = checkbox
            band_layout.addWidget(checkbox)
        
        band_group.setLayout(band_layout)
        
        # Output file selection
        output_group = QGroupBox("Output File")
        output_layout = QHBoxLayout()
        
        self.output_path = QLineEdit()
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        default_file = os.path.join(default_dir, "results.txt")
        self.output_path.setText(default_file)
        
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_file)
        
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(browse_button)
        output_group.setLayout(output_layout)
        
        # Terminal/console output
        terminal_group = QGroupBox("Console Output")
        terminal_layout = QVBoxLayout()
        
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setFont(QFont("Courier", 10))
        self.terminal.setStyleSheet("background-color: black; color: white;")
        
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
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_script)
        self.cancel_button.setEnabled(False)
        
        exit_button = QPushButton("Exit")
        exit_button.clicked.connect(self.close)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch(1)
        button_layout.addWidget(exit_button)
        
        # Add components to main layout
        main_layout.addWidget(framework_group)
        main_layout.addWidget(modality_group)
        main_layout.addWidget(feature_method_group)
        main_layout.addWidget(band_group)
        main_layout.addWidget(output_group)
        main_layout.addWidget(terminal_group)
        main_layout.addWidget(self.progress_bar)
        main_layout.addLayout(button_layout)
        
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
    
    def browse_file(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Output File", self.output_path.text(),
            "Text Files (*.txt);;All Files (*)")
        if file_path:
            self.output_path.setText(file_path)
            self.statusBar.showMessage(f'Output file set to {file_path}')
    
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
        
        # Get selected options
        framework = "tensorflow" if self.tf_radio.isChecked() else "pytorch"
        modalities = self.get_selected_modalities()
        feature_methods = self.get_selected_feature_methods()
        selected_bands = self.get_selected_bands()
        output_file = self.output_path.text()
        
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
        self.terminal.append(f"Output will be saved to: {output_file}\n\n")
        
        # Update UI state
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.statusBar.showMessage(f'Running {framework.capitalize()} implementation...')
        self.running = True
        
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
                "--output", output_file
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
            
            # Capture and redirect output
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in self.process.stdout:
                    if line:
                        print(line, end='')  # This will go through our redirector
                        f.write(line)
                        f.flush()
            
            return_code = self.process.wait()
            if return_code == 0:
                self.update_status(f"Script completed successfully: {os.path.basename(output_file)}")
            else:
                self.update_status(f"Script exited with code {return_code}: {os.path.basename(output_file)}")
                
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
        finally:
            # Check if all threads are completed
            active_threads = [t for t in self.execution_threads if t.is_alive()]
            if not active_threads:
                # Reset stdout/stderr only when all threads are done
                sys.stdout = self.original_stdout
                sys.stderr = self.original_stderr
                
                # Update UI state
                self.update_ui_after_execution()
    
    def update_status(self, message):
        """Update status bar with message (thread-safe)."""
        QApplication.postEvent(self, 
                               StatusUpdateEvent(message))
    
    def update_ui_after_execution(self):
        """Reset UI elements after script execution (thread-safe)."""
        QApplication.postEvent(self, UIUpdateEvent())
    
    def cancel_script(self):
        """Cancel the running script."""
        if self.process and self.running:
            self.process.terminate()
            self.terminal.append("\n\n*** Script execution canceled by user ***\n")
            self.update_status("Execution canceled")
            
            # No need to update UI here as it will be handled
            # when the process terminates in the run_script method
    
    def handle_ui_update(self):
        """Handle UI updates after script execution."""
        self.start_button.setEnabled(True)
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


from PyQt5.QtCore import QEvent

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
