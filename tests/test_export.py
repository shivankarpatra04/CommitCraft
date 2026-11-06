"""GGUF upload and the export script, with the Hub mocked. No network or model needed."""

from unittest.mock import MagicMock, patch

import pytest

import config
from train import export


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_TOKEN", "test-value")
    api = MagicMock()
    api.list_repo_files.return_value = [config.GGUF_FILENAME]
    with patch("huggingface_hub.HfApi", lambda token: api):
        yield api


def test_upload_sends_the_gguf_to_the_model_repo(client: MagicMock) -> None:
    export.upload()
    upload = client.upload_file.call_args.kwargs
    assert upload["repo_id"] == config.ADAPTER_REPO
    assert upload["path_in_repo"] == config.GGUF_FILENAME


def test_upload_fails_when_the_hub_does_not_have_the_file(client: MagicMock) -> None:
    client.list_repo_files.return_value = ["README.md"]
    with pytest.raises(SystemExit):
        export.upload()


def test_export_script_stops_on_errors_and_quantizes_to_q4_k_m() -> None:
    script = (config.ROOT_DIR / "scripts" / "export_gguf.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in script
    assert "Q4_K_M" in script
    assert script.index("export merge") < script.index("export upload")
