"""Local music analysis app: stems, notes, drums, and tempo."""

import logging


class _QuietOptionalBackends(logging.Filter):
    """Hide Basic Pitch hints about ML backends this app does not use."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "reinstall basic-pitch" not in message


logging.getLogger().addFilter(_QuietOptionalBackends())
