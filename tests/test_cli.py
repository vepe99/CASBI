from CASBI.__main__ import main

from click.testing import CliRunner


def test_CASBI_cli():
    runner = CliRunner()
    result = runner.invoke(main, ())
    assert result.exit_code == 0
