"""Constantes do domínio de portfólios."""

from decimal import Decimal

# Precisão de arredondamento dos pesos da distribuição da carteira.
WEIGHT_PRECISION = Decimal('0.000001')

# Janela (em semanas) usada para a variação semanal no cálculo de PnL.
PNL_WEEK_LOOKBACK = 1
