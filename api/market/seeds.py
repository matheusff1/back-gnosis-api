"""Definições canônicas para semear o catálogo de ativos.

Fonte única de verdade usada pelo comando ``seed_assets`` e também pelo
``DataCollector`` (que passa a obter os símbolos a coletar a partir de
``Asset``/``AssetSource`` no banco, em vez de listas hardcoded).

Formato de ``ASSET_SEED``: ``symbol -> (asset_type, data_source, source_symbol, extra)``
- ``asset_type``: chave em ``ASSET_TYPES``.
- ``data_source``: chave em ``DATA_SOURCES``.
- ``source_symbol``: código do ativo naquele provedor.
- ``extra``: parâmetros específicos da fonte (ex.: número da série SGS do BACEN).
"""

ASSET_TYPES = {
    'stock': 'Ação',
    'currency': 'Moeda',
    'crypto': 'Criptomoeda',
    'index': 'Índice',
    'commodity': 'Commodity / Futuro',
    'bond': 'Título / Renda fixa',
    'economic_indicator': 'Indicador econômico',
    'unknown': 'Não classificado',
}

DATA_SOURCES = {
    'yfinance': 'Yahoo Finance',
    'fred': 'Federal Reserve Economic Data (FRED)',
    'bacen': 'Banco Central do Brasil - SGS',
    'twelvedata': 'Twelve Data',
}

# Tipo atribuído a símbolos presentes no banco mas ausentes do ASSET_SEED.
FALLBACK_TYPE = 'unknown'

ASSET_SEED = {
    # --- Ações brasileiras ---
    'PETR4.SA': ('stock', 'yfinance', 'PETR4.SA', {}),
    'VALE3.SA': ('stock', 'yfinance', 'VALE3.SA', {}),
    'ITUB4.SA': ('stock', 'yfinance', 'ITUB4.SA', {}),
    'BBDC4.SA': ('stock', 'yfinance', 'BBDC4.SA', {}),
    'ABEV3.SA': ('stock', 'yfinance', 'ABEV3.SA', {}),
    'WEGE3.SA': ('stock', 'yfinance', 'WEGE3.SA', {}),
    'B3SA3.SA': ('stock', 'yfinance', 'B3SA3.SA', {}),
    'ITSA4.SA': ('stock', 'yfinance', 'ITSA4.SA', {}),
    'CSAN3.SA': ('stock', 'yfinance', 'CSAN3.SA', {}),
    'BRFS': ('stock', 'yfinance', 'BRFS', {}),

    # --- Ações americanas ---
    'AAPL': ('stock', 'yfinance', 'AAPL', {}),
    'NVDA': ('stock', 'yfinance', 'NVDA', {}),
    'MSFT': ('stock', 'yfinance', 'MSFT', {}),
    'AMZN': ('stock', 'yfinance', 'AMZN', {}),
    'GOOGL': ('stock', 'yfinance', 'GOOGL', {}),
    'META': ('stock', 'yfinance', 'META', {}),
    'TSLA': ('stock', 'yfinance', 'TSLA', {}),
    'UNH': ('stock', 'yfinance', 'UNH', {}),
    'DHR': ('stock', 'yfinance', 'DHR', {}),
    'SPGI': ('stock', 'yfinance', 'SPGI', {}),

    # --- Moeda / Cripto ---
    'BRL=X': ('currency', 'yfinance', 'BRL=X', {}),
    'BTC-USD': ('crypto', 'yfinance', 'BTC-USD', {}),

    # --- Índices ---
    '^BVSP': ('index', 'yfinance', '^BVSP', {}),
    '^GSPC': ('index', 'yfinance', '^GSPC', {}),
    '^FVX': ('index', 'yfinance', '^FVX', {}),
    'GSG': ('index', 'yfinance', 'GSG', {}),

    # --- Commodities / futuros ---
    'CL=F': ('commodity', 'yfinance', 'CL=F', {}),
    'BZ=F': ('commodity', 'yfinance', 'BZ=F', {}),
    'NG=F': ('commodity', 'yfinance', 'NG=F', {}),
    'ZC=F': ('commodity', 'yfinance', 'ZC=F', {}),
    'ZS=F': ('commodity', 'yfinance', 'ZS=F', {}),
    'GC=F': ('commodity', 'yfinance', 'GC=F', {}),
    'DX=F': ('commodity', 'yfinance', 'DX=F', {}),

    # --- Títulos / renda fixa (ETFs/fundos) ---
    'NTNS11.SA': ('bond', 'yfinance', 'NTNS11.SA', {}),
    'BTGIMABFIRF.SA': ('bond', 'yfinance', 'BTGIMABFIRF.SA', {}),

    # --- Indicadores econômicos (FRED) ---
    'M2SL': ('economic_indicator', 'fred', 'M2SL', {}),
    'NFCI': ('economic_indicator', 'fred', 'NFCI', {}),
    'TEDRATE': ('economic_indicator', 'fred', 'TEDRATE', {}),

    # --- Indicadores econômicos (BACEN - SGS) ---
    'CDI': ('economic_indicator', 'bacen', 'CDI', {'sgs_series': 11}),
    'Swap_DI_5Y': ('economic_indicator', 'bacen', 'Swap_DI_5Y', {'sgs_series': 1178}),
    'Selic_Over': ('economic_indicator', 'bacen', 'Selic_Over', {}),
    'Selic_Over_Long': ('economic_indicator', 'bacen', 'Selic_Over_Long', {}),
}
