"""Agregador de modelos do app `api`.

Os modelos vivem em cada domínio (`api/market`, `api/quant`, `api/portfolios`),
mas continuam registrados sob o app_label `api`. Este módulo re-exporta todos
eles para que o Django os descubra (ele importa `api.models`) e para manter
compatível qualquer `from api.models import X` já existente no projeto.
"""

from .market.models import (
    Asset,
    AssetSource,
    AssetType,
    DataSource,
    MarketData,
)
from .quant.models import (
    Prediction,
    PredictionFeature,
    PredictionPoint,
    PredictionTrainingEpoch,
)
from .portfolios.models import (
    Portfolio,
    PortfolioAsset,
    PortfolioConfig,
    PortfolioTracking,
    PortfolioTrackingAsset,
)

__all__ = [
    'MarketData',
    'DataSource',
    'AssetType',
    'Asset',
    'AssetSource',
    'Prediction',
    'PredictionPoint',
    'PredictionFeature',
    'PredictionTrainingEpoch',
    'Portfolio',
    'PortfolioAsset',
    'PortfolioConfig',
    'PortfolioTracking',
    'PortfolioTrackingAsset',
]
