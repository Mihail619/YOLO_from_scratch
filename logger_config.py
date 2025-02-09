from loguru import logger

logger.add("logging/info.log", format="{time} {level} {message}", level="INFO", encoding="utf-8", rotation="1000KB")
logger.add("logging/debug.log", format="{time} {level} {message}", level="DEBUG", encoding="utf-8", rotation="1000KB")
logger.add("logging/error.log", format="{time} {level} {message}", level="ERROR", encoding="utf-8")
