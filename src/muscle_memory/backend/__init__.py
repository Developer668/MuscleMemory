"""Production composition for the Muscle Memory service backend."""

from muscle_memory.backend.api_backend import MuscleMemoryApiBackend
from muscle_memory.backend.config import BackendConfig
from muscle_memory.backend.episode_journal import CoordinatorEpisodeJournal
from muscle_memory.backend.episode_runtime import OperationalEpisodeRuntime
from muscle_memory.backend.providers import (
    ProviderDeployment,
    ProviderEvidence,
    ProviderReadiness,
    ProviderRegistry,
)

__all__ = [
    "BackendConfig",
    "CoordinatorEpisodeJournal",
    "MuscleMemoryApiBackend",
    "OperationalEpisodeRuntime",
    "ProviderDeployment",
    "ProviderEvidence",
    "ProviderReadiness",
    "ProviderRegistry",
]
