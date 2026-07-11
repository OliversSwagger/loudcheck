from enum import Enum

class ReadinessStatus(str, Enum):
    READY = "ready"
    CONDITIONALLY_READY = "conditionally_ready"
    NOT_READY = "not_ready"
