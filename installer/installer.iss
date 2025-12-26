[Setup]
AppName=MoodyStream
AppVersion=1.0.0
DefaultDirName={pf}\MoodyStream
DefaultGroupName=MoodyStream
OutputBaseFilename=MoodyStreamSetup
Compression=lzma2
SolidCompression=yes
OutputDir=Output

[Files]
Source: ""..\dist\MoodyStream\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\MoodyStream"; Filename: "{app}\MoodyStream.exe"
Name: "{commondesktop}\MoodyStream"; Filename: "{app}\MoodyStream.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Desktop-Icon erstellen"; GroupDescription: "Zusätzliche Aufgaben:"
