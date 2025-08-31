[Setup]
AppName=KeilProc
AppVersion=0.1.0
AppVerName=KeilProc 0.1.0
DefaultDirName="{pf}\KeilProc"
DisableDirPage=no
DefaultGroupName=KeilProc
OutputBaseFilename=KeilProcInstaller
Compression=lzma
SolidCompression=yes

[Files]
Source: "..\..\kielproc_monorepo\dist\KeilProc\KeilProc.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\KeilProc"; Filename: "{app}\KeilProc.exe"
Name: "{commondesktop}\KeilProc"; Filename: "{app}\KeilProc.exe"; Tasks: desktopicon

[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"
