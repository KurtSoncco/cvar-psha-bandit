"""CVaR policy learning for PSHA logic-tree Importance Sampling."""


def main(argv: list[str] | None = None) -> None:
    from cvar_psha.experiment.cli import main as _main

    _main(argv)


__all__ = ["main"]
