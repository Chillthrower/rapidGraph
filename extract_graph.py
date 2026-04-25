import rapidgraph.core as _core
from rapidgraph.core import *  # noqa: F401,F403


def main(argv=None):
    _core.build_default_extractor = build_default_extractor
    _core.export_graph_to_neo4j = export_graph_to_neo4j
    return _core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
