[Setup]
AppName=KielProc
AppVersion=0.1.0
AppVerName=KielProc 0.1.0
DefaultDirName="{pf}\KielProc"
DisableDirPage=no
DefaultGroupName=KielProc
OutputBaseFilename=KielProcInstaller
Compression=lzma
SolidCompression=yes

[Files]
Source: "..\..\dist\RunEasy-GUI.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\RunEasy"; Filename: "{app}\RunEasy-GUI.exe"
Name: "{commondesktop}\RunEasy"; Filename: "{app}\RunEasy-GUI.exe"; Tasks: desktopicon

[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"
