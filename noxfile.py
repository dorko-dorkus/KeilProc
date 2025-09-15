import nox
from pathlib import Path

ROOT = Path(__file__).parent

@nox.session
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    session.install(
        "-r",
        "requirements.txt",
        "-c",
        "constraints.txt",
        "pytest",
    )
    session.install("-e", ".")
    session.run("pytest")

@nox.session
def smoke(session: nox.Session) -> None:
    """Ensure the package imports in a clean environment."""
    session.install(
        "-r",
        "requirements.txt",
        "-c",
        "constraints.txt",
    )
    session.install("-e", ".")
    session.run("python", "-c", "import kielproc")
