"""Command line: ``ute prepare | train | report | serve``."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ute", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prepare", help="download the dataset and convert it to parquet")

    t = sub.add_parser("train", help="fit, evaluate and save the production model")
    t.add_argument("--sample-frac", type=float, default=None, help="e.g. 0.1 for a quick run")
    t.add_argument("--outcomes", nargs="+", default=["conversion", "visit"])
    t.add_argument("--learners", nargs="+", default=None)
    t.add_argument("--n-boot", type=int, default=200)
    t.add_argument("--no-leakage", action="store_true")

    sub.add_parser("report", help="regenerate docs/results.md and the README results block")

    s = sub.add_parser("serve", help="run the scoring API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    a = p.parse_args(argv)
    if a.cmd == "prepare":
        from ute import data

        print("downloading ->", data.download())
        print("parquet ->", data.to_parquet())
    elif a.cmd == "train":
        from ute import learners, train

        train.run(
            sample_frac=a.sample_frac,
            outcomes=tuple(a.outcomes),
            learners=tuple(a.learners or learners.LEARNERS),
            n_boot=a.n_boot,
            leakage=not a.no_leakage,
        )
    elif a.cmd == "report":
        from ute import report

        report.main()
    elif a.cmd == "serve":
        import uvicorn

        uvicorn.run("ute.api:app", host=a.host, port=a.port)


if __name__ == "__main__":
    main()
