import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QCloseEvent

from setup.setup_wizard import SetupController
from gui.main_window import MainWindow


# ----------------------------------------------------------------------
#  Helper: Check if setup is complete
# ----------------------------------------------------------------------
def check_setup_complete():
    """Check if setup is complete by verifying 3 JSON files exist."""

    required_files = [
        "setup/rest_face_model.json",
        "setup/setup_config.json",
        "setup/rest_face_model.profiles.snapshot.json"
    ]

    print("🔍 Checking setup files...")

    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ Found: {file_path}")
        else:
            print(f"  ❌ Missing: {file_path}")
            all_exist = False

    return all_exist


# ----------------------------------------------------------------------
#  Main Controller Class
# ----------------------------------------------------------------------
class AppController:
    """
    Central application controller.
    Manages switching between setup wizard and main window.
    """

    def __init__(self):
        self.app = QApplication(sys.argv)
        
        # CRITICAL: Don't quit when the last window closes
        # This allows us to close setup windows and open main window without app exiting
        self.app.setQuitOnLastWindowClosed(False)

        self.main_window = None
        self.settings_dialog = None
        self.wizard_controller = None

        print("🚀 Starting MoodyStream...")

        # Decide what to show first
        if check_setup_complete():
            print("\n✅ Setup complete → Opening main window.")
            self.open_main_window()
        else:
            print("\n📋 Setup incomplete → Opening setup wizard.")
            self.open_setup_wizard()

    # ------------------------------------------------------------------
    #  WINDOW MANAGEMENT
    # ------------------------------------------------------------------

    def open_setup_wizard(self):
        """Open Setup Wizard and clean up MainWindow if needed."""
        print("🧙 Opening Setup Wizard...")

        # Close settings dialog if open
        if self.settings_dialog is not None:
            print("🛑 Closing Settings Dialog...")
            self.settings_dialog.close()
            self.settings_dialog = None

        # Properly close main window using closeEvent
        if self.main_window is not None:
            print("🛑 Properly closing MainWindow before wizard restart...")
            # Set flag to prevent double closeEvent
            self.main_window._is_closing = True
            # Stop detection manually without calling closeEvent
            self.main_window.stop_detection_internal()
            # Now close the window (closeEvent will skip cleanup due to flag)
            self.main_window.close()
            self.main_window = None
            print("✅ MainWindow closed cleanly")
            
            # IMPORTANT: Give the camera time to fully release
            import time
            print("⏳ Waiting for camera to release...")
            time.sleep(0.5)  # Wait 500ms for camera to fully release
            self.app.processEvents()  # Process any pending events

        # Create wizard controller
        self.wizard_controller = SetupController()
        # Connect signal when setup is finished
        self.wizard_controller.finished.connect(self._on_setup_finished)

        # Start the setup wizard
        self.wizard_controller.start()

    def _on_setup_finished(self, success: bool):
        """Handle Setup Wizard finish."""
        if success:
            print("✅ Setup finished successfully!")
            self.open_main_window()
        else:
            print("❌ Setup cancelled or failed.")
            # Optionally: reopen wizard or exit
            self.app.quit()

    def open_main_window(self):
        """Open MainWindow and connect restart signal."""
        print("🎭 Opening Main Window...")

        # Close wizard if still open
        if self.wizard_controller is not None:
            self.wizard_controller = None

        # Create / show main window
        self.main_window = MainWindow(camera_index=0)

        # Save reference to settings dialog if needed
        self.main_window.settings_dialog = None

        # IMPORTANT: connect restart setup
        self.main_window.restart_setup_signal.connect(self.open_setup_wizard)

        # Override MainWindow.open_settings to capture the settings dialog
        original_open_settings = self.main_window.open_settings

        def open_settings_override():
            original_open_settings()
            self.settings_dialog = self.main_window.findChild(type(self.main_window.settings_dialog), "")
            # Fallback if not found
            if self.settings_dialog is None:
                print("⚠️ Settings dialog reference not captured")
        self.main_window.open_settings = open_settings_override

        self.main_window.show()
        
        # CRITICAL: Ensure the window stays visible and processes events
        self.app.processEvents()
        print("✅ Main Window is now active and visible")

    # ------------------------------------------------------------------
    #  Start Event Loop
    # ------------------------------------------------------------------
    def run(self):
        return self.app.exec_()


# ----------------------------------------------------------------------
#  Program Entry Point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    controller = AppController()
    sys.exit(controller.run())