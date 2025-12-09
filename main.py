"""
main.py - Application entry point

Workflow:
1. Run setup wizard (if needed)
2. Open main window
3. User clicks "Detection: OFF" button to start camera
"""

import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from setup.setup_wizard import main as run_setup_wizard
from gui.main_window import MainWindow


def check_setup_complete():
    """
    Check if setup is complete by verifying 3 JSON files exist.
    
    Returns:
        bool: True if all 3 JSON files exist, False otherwise
    """
    # Define the 3 required JSON files
    required_files = [
        "setup/rest_face_model.json",
        "setup/setup_config.json", 
        "setup/rest_face_model.profiles.snapshot.json.json"
    ]
    
    print("🔍 Checking setup files...")
    
    all_exist = True
    for file_path in required_files:
        full_path = Path(file_path)
        if full_path.exists():
            print(f"  ✅ Found: {file_path}")
        else:
            print(f"  ❌ Missing: {file_path}")
            all_exist = False
    
    return all_exist


def main():
    """Main entry point"""
    # Create QApplication early (required for GUI)
    app = QApplication(sys.argv)
    
    print("🚀 Starting MoodyStream...")
    
    # Check if setup is complete
    setup_complete = check_setup_complete()
    
    if not setup_complete:
        print("\n📋 Setup incomplete - launching setup wizard...")
        # start setup
        setup_wizard_success = run_setup_wizard()
        if not setup_wizard_success:
            print("❌ SetupWizard abgebrochen oder fehlgeschlagen.")
            return
    else:
        print("\n✅ Setup complete - all files found!")
    
    # Launch main window
    print("\n🎭 Opening main window...")
    
    window = MainWindow(camera_index=0)
    window.show()
        
    print("✅ Main window opened")
        
    # Run event loop
    return app.exec_()


if __name__ == "__main__":
     main()