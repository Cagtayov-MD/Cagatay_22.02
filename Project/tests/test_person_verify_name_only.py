import os
import sys

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


class _FakeTMDB:
    def search_person(self, name):
        return [{
            "id": 123,
            "name": "Alice Rohrwacher",
            "known_for_department": "directing",
        }]


def test_person_verify_returns_name_only_without_mapped_role():
    from core.person_verify import PersonVerifier

    verifier = PersonVerifier(tmdb_client=_FakeTMDB())
    result = verifier.verify_name("ALICE BOHKWACHEB")

    assert result["found"] is True
    assert result["name"] == "Alice Rohrwacher"
    assert result["known_for_department"] == "directing"
    assert "mapped_role" not in result


def test_person_verify_non_person_filter_still_works():
    from core.person_verify import PersonVerifier

    verifier = PersonVerifier(tmdb_client=_FakeTMDB())
    result = verifier.verify_name("WITH THE ASSISTANCE OF THE")

    assert result["found"] is False
    assert result["source"] == "non_person_filter"
