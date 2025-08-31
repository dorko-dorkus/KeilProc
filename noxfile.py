import nox
from pathlib import Path

ROOT = Path(__file__).parent
MONOREPO = ROOT / "kielproc_monorepo"

@nox.session
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    session.install(
        "-r", str(MONOREPO / "requirements.txt"),
        "-c", str(MONOREPO / "constraints.txt"),
        "pytest",
    )
    session.chdir(str(MONOREPO))
    session.run("pytest")

@nox.session
def smoke(session: nox.Session) -> None:
    """Exercise the CLI help to ensure the package imports."""
    session.install(
        "-r", str(MONOREPO / "requirements.txt"),
        "-c", str(MONOREPO / "constraints.txt"),
    )
    session.chdir(str(MONOREPO))
    session.run("python", "-m", "kielproc.cli", "--help")
