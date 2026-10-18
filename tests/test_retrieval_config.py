from copilot import Config
from copilot.retrieval import Retriever, load_corpus, tokenize


def test_corpus_loads_with_unique_id_version_pairs():
    docs = load_corpus()
    assert len(docs) >= 20
    assert len({(d.doc_id, d.version) for d in docs}) == len(docs)


def test_superseded_versions_exist():
    versions = sorted(d.version for d in load_corpus() if d.doc_id == "billing-policy")
    assert versions == [1, 2]


def test_latest_only_excludes_superseded(retriever):
    hits = retriever.search("refund window days", k=10)
    bp = [p for p in hits if p.doc_id == "billing-policy"]
    assert bp and all(p.version == 2 for p in bp)
    assert "30 days" in hits[0].text


def test_including_superseded_surfaces_v1():
    r = Retriever(latest_only=False)
    hits = r.search("refund window days", k=10)
    assert {p.version for p in hits if p.doc_id == "billing-policy"} == {1, 2}
    assert any("14 days" in p.text for p in hits)


def test_area_filter(retriever):
    hits = retriever.search("rate limit key", k=10, area="api")
    assert hits and all(p.area == "api" for p in hits)
    assert not any(p.doc_id == "password-reset" for p in retriever.search("password reset", area="billing"))


def test_results_are_sorted_and_deterministic(retriever):
    a = retriever.search("how do I reset my password", k=5)
    b = Retriever().search("how do I reset my password", k=5)
    assert [(p.doc_id, p.section, p.score) for p in a] == [(p.doc_id, p.section, p.score) for p in b]
    assert [p.score for p in a] == sorted((p.score for p in a), reverse=True)
    assert a[0].doc_id == "password-reset"


def test_passage_carries_citation_metadata(retriever):
    p = retriever.search("snapshot retention backups", k=1)[0]
    assert p.doc_id == "backup-restore" and p.section and p.score > 0 and p.area == "storage"


def test_nonsense_query_has_no_hits(retriever):
    assert retriever.search("zxqv plorp", k=3) == []
    assert retriever.search("the and of", k=3) == []


def test_tokenizer_drops_stopwords_and_stems():
    assert tokenize("The refunds are in the Queue") == ["refund", "queue"]


def test_injection_documents_are_in_the_corpus(retriever):
    assert "Ignore previous instructions" in retriever.doc_text("legacy-migration-notes")
    assert "SYSTEM:" in retriever.doc_text("community-tips")


def test_config_hash_tracks_retriever_and_prompt_settings_only():
    base = Config()
    assert base.config_hash() == Config().config_hash()
    assert base.with_(bm25_k1=1.2).config_hash() != base.config_hash()
    assert base.with_(prompt_version="support-prompt-v4").config_hash() != base.config_hash()
    assert base.with_(latest_only=False).config_hash() != base.config_hash()
    assert base.with_(max_steps=3).config_hash() == base.config_hash()
