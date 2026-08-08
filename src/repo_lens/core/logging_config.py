import logging

import structlog

NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "mcp",
    "chromadb",
    "google_genai",
    "anthropic",
    "asyncio",
    # Chainlit / uvicorn stack (created after module import)
    "engineio",
    "socketio",
    "uvicorn",
    "chainlit",
    "watchfiles",
)


def configure_logging(log_level: str = "INFO") -> None:
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.ExtraAdder(),
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(),
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level.upper())

    noisy_level = logging.DEBUG if log_level.upper() == "DEBUG" else logging.WARNING

    for prefix in NOISY_LOGGERS:
        lib_logger = logging.getLogger(prefix)
        lib_logger.handlers.clear()
        lib_logger.setLevel(noisy_level)

    for name in logging.root.manager.loggerDict:
        lib_logger = logging.getLogger(name)
        lib_logger.handlers.clear()
        if any(name.startswith(prefix) for prefix in NOISY_LOGGERS):
            lib_logger.setLevel(noisy_level)
        else:
            lib_logger.setLevel(logging.NOTSET)
