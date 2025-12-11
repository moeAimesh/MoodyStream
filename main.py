import sys
import time
from pathlib import Path
from PyQt5.QtWidgets import QApplication

from setup.setup_wizard import SetupController
from gui.main_window import MainWindow

# check if setup is already complete
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

#  main controller class
class AppController:
    """
    Central application controller.
    Manages switching between setup wizard and main window.
    """

    def __init__(self):
        self.app = QApplication(sys.argv)
        
        # switch between windwos
        self.app.setQuitOnLastWindowClosed(False)

        self.main_window = None
        self.settings_dialog = None
        self.wizard_controller = None

        print("🚀 Starting MoodyStream...")

        # decide what comes firs - setup or main?
        if check_setup_complete():
            print("\n✅ Setup complete → Opening main window.")
            self.open_main_window()
        else:
            print("\n📋 Setup incomplete → Opening setup wizard.")
            self.open_setup_wizard()

    def open_setup_wizard(self):
        """Open Setup Wizard and clean up MainWindow if needed."""
        print("🧙 Opening Setup Wizard...")

        # Properly close main window using closeevent
        if self.main_window is not None:
            print("🛑 Properly closing MainWindow before wizard restart...")
            # set flag to prevent double closeevent
            self.main_window._is_closing = True
            # stop detection manually without calling closeevent
            self.main_window.stop_detection_internal()
            # now close the window (closeevent will skip cleanup due to flag)
            self.main_window.close()
            self.main_window = None
            print("✅ MainWindow closed cleanly")
            
            print("⏳ Waiting for camera to release...")
            time.sleep(0.5)  # Wait 500ms for camera to fully release
            self.app.processEvents()  # Process any pending events

        # create wizard controller
        self.wizard_controller = SetupController()
        # connect signal when setup is finished
        self.wizard_controller.finished.connect(self._on_setup_finished)

        # start the setup wizard
        self.wizard_controller.start()

    def _on_setup_finished(self, success: bool):
        """Handle Setup Wizard finish."""
        if success:
            print("✅ Setup finished successfully!")
            self.open_main_window()
        else:
            print("❌ Setup cancelled or failed.")
            self.app.quit()

    def open_main_window(self):
        """Open MainWindow and connect restart signal."""
        print("🎭 Opening Main Window...")

        # close wizard if still open
        if self.wizard_controller is not None:
            self.wizard_controller = None

        # create / show main window
        self.main_window = MainWindow(camera_index=0)

        # connect restart setup
        self.main_window.restart_setup_signal.connect(self.open_setup_wizard)

        self.main_window.show()
        
        # ensure the window stays visible and processes events
        self.app.processEvents()
        print("✅ Main Window is now active and visible")

    # run app
    def run(self):
        return self.app.exec_()


if __name__ == "__main__":
    controller = AppController()
    sys.exit(controller.run())