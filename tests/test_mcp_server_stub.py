"""The `precedent-mcp` stub prints one line and exits 2, with no traceback."""

from precedent import mcp_server
from precedent.cli import main as precedent_main


def test_run_and_main_report_not_built_and_return_2(capsys):
    assert mcp_server.run([]) == 2
    assert mcp_server.main([]) == 2
    captured = capsys.readouterr()
    assert captured.err == "not built yet: mcp\nnot built yet: mcp\n"
    assert captured.out == ""


def test_the_precedent_mcp_subcommand_reaches_the_stub(capsys):
    assert precedent_main(["mcp"]) == 2
    assert capsys.readouterr().err == "not built yet: mcp\n"
