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
Source: "..\..\dist\kielproc-gui\kielproc-gui.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\KielProc GUI"; Filename: "{app}\kielproc-gui.exe"
Name: "{commondesktop}\KielProc GUI"; Filename: "{app}\kielproc-gui.exe"; Tasks: desktopicon

[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"
