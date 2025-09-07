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
Source: "..\..\app\dist\KielProc\KielProc.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\KielProc"; Filename: "{app}\KielProc.exe"
Name: "{commondesktop}\KielProc"; Filename: "{app}\KielProc.exe"; Tasks: desktopicon

[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"
