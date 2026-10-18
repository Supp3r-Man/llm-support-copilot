import pytest

from copilot import Config, build_agent
from copilot.retrieval import Retriever
from copilot.server import make_server
from copilot.tools import ToolRegistry


@pytest.fixture(scope="session")
def cfg():
    return Config()


@pytest.fixture(scope="session")
def retriever(cfg):
    return Retriever.from_config(cfg)


@pytest.fixture
def registry(cfg, retriever):
    return ToolRegistry(cfg, retriever)  # fresh refund/audit state per test


@pytest.fixture
def agent_factory(tmp_path):
    def make(**kw):
        kw.setdefault("trace_dir", tmp_path / "traces")
        return build_agent(**kw)

    return make


@pytest.fixture
def server_factory():
    servers = []

    def make(**kw):
        s = make_server(**kw).start()
        servers.append(s)
        return s

    yield make
    for s in servers:
        try:
            s.shutdown()
        except Exception:
            pass


@pytest.fixture(scope="session")
def eval_runs():
    """Every variant run once over the full golden set (deterministic, ~10 s); shared by the eval tests."""
    from copilot_eval.report import run_all

    return run_all("all")
