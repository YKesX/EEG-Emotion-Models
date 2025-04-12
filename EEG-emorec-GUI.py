#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EEG Emotion Recognition Models GUI Application
This application provides a graphical interface to:
- Select between TensorFlow and PyTorch implementations
- Choose an output directory for results
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
    QTextEdit, QFileDialog, QGroupBox, QProgressBar, QMessageBox, QStatusBar
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
        
        # Output directory selection
        output_group = QGroupBox("Output Directory")
        output_layout = QHBoxLayout()
        
        self.output_path = QLineEdit()
        default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        self.output_path.setText(default_path)
        
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_directory)
        
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
    
    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.output_path.text())
        if directory:
            self.output_path.setText(directory)
            self.statusBar.showMessage(f'Output directory set to {directory}')
    
    @pyqtSlot(str)
    def update_terminal(self, text):
        """Update the terminal with new output."""
        self.terminal.moveCursor(QTextCursor.End)
        self.terminal.insertPlainText(text)
        self.terminal.moveCursor(QTextCursor.End)
    
    def start_script(self):
        """Start execution of the selected script."""
        if self.running:
            return
        
        # Get selected framework and output path
        framework = "tensorflow" if self.tf_radio.isChecked() else "pytorch"
        output_dir = self.output_path.text()
        
        # Ensure output directory exists
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
        self.terminal.append(f"Output will be saved to: {output_dir}\n\n")
        
        # Update UI state
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.statusBar.showMessage(f'Running {framework.capitalize()} implementation...')
        self.running = True
        
        # Redirect stdout/stderr
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stderr_redirector
        
        # Start execution in a separate thread
        self.execution_thread = threading.Thread(
            target=self.run_script, 
            args=(script_path, output_dir)
        )
        self.execution_thread.daemon = True
        self.execution_thread.start()
    
    def run_script(self, script_path, output_dir):
        """Execute the script in a subprocess."""
        try:
            # Create output file path
            output_file = os.path.join(output_dir, "results.txt")
            
            # Prepare the command
            cmd = [sys.executable, script_path]
            
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
                self.update_status("Script completed successfully")
            else:
                self.update_status(f"Script exited with code {return_code}")
                
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
        finally:
            # Reset stdout/stderr
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
        if event.type() == UIUpdateEvent.EVENT_TYPE:
            self.handle_ui_update()
            return True
        return super().event(event)
        
    def customEvent(self, event):
        if event.type() == UIUpdateEvent.EVENT_TYPE:
            self.handle_ui_update()
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
