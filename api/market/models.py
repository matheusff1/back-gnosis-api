from django.db import models


class DataSource(models.Model):
    """Fonte de dados de mercado (ex.: yfinance, FRED, BACEN, TwelveData)."""
    name = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True, null=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        app_label = 'api'
        db_table = 'data_source'
        ordering = ['name']
        verbose_name = 'Data Source'
        verbose_name_plural = 'Data Sources'

    def __str__(self):
        return self.name


class AssetType(models.Model):
    """Tipo do ativo (ação, moeda, cripto, índice, commodity, título, indicador)."""
    name = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True, null=True)

    class Meta:
        app_label = 'api'
        db_table = 'asset_type'
        ordering = ['name']
        verbose_name = 'Asset Type'
        verbose_name_plural = 'Asset Types'

    def __str__(self):
        return self.name


class Asset(models.Model):
    """Ativo negociável ou série usada como feature nos modelos.

    ``symbol`` é o código canônico (o mesmo já armazenado em ``MarketData``).
    ``is_allowed`` marca os ativos expostos ao front (universo negociável);
    no back-end todos os ativos são usados (ex.: modelos de risco/predição).
    """
    symbol = models.CharField(max_length=30, unique=True)
    name = models.CharField(max_length=150, blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    asset_type = models.ForeignKey(
        AssetType, on_delete=models.PROTECT, related_name='assets'
    )
    is_allowed = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    data_sources = models.ManyToManyField(
        DataSource, through='AssetSource', related_name='assets'
    )

    class Meta:
        app_label = 'api'
        db_table = 'asset'
        ordering = ['symbol']
        verbose_name = 'Asset'
        verbose_name_plural = 'Assets'

    def __str__(self):
        return self.symbol


class AssetSource(models.Model):
    """Vínculo ativo↔fonte: código do ativo em cada provedor.

    Permite que o mesmo ativo tenha fontes diferentes (ex.: yfinance 'PETR4.SA'
    vs TwelveData 'PETR4'), e guarda parâmetros específicos em ``extra`` (ex.: o
    número da série SGS do BACEN).
    """
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE, related_name='sources')
    data_source = models.ForeignKey(
        DataSource, on_delete=models.PROTECT, related_name='asset_links'
    )
    source_symbol = models.CharField(max_length=50)
    extra = models.JSONField(default=dict, blank=True)
    is_primary = models.BooleanField(default=True)

    class Meta:
        app_label = 'api'
        db_table = 'asset_source'
        unique_together = ('asset', 'data_source')
        verbose_name = 'Asset Source'
        verbose_name_plural = 'Asset Sources'

    def __str__(self):
        return f"{self.asset.symbol} @ {self.data_source.name} ({self.source_symbol})"


# COLUNAS DE MARKET DATA: Date	Close	High	Low	Open	Volume	Symbol
class MarketData(models.Model):
    date = models.DateField()
    close = models.DecimalField(max_digits=20, decimal_places=2)
    high = models.DecimalField(max_digits=20, decimal_places=2)
    low = models.DecimalField(max_digits=20, decimal_places=2)
    open = models.DecimalField(max_digits=20, decimal_places=2)
    volume = models.BigIntegerField()
    # Sempre populado na prática; nullable no schema só para evitar migração
    # interativa. O writer garante o vínculo e ignora linhas sem asset.
    asset = models.ForeignKey(
        Asset,
        on_delete=models.PROTECT,
        related_name='market_data',
        null=True,
        blank=True,
    )

    class Meta:
        app_label = 'api'
        db_table = 'market_data'
        unique_together = ('date', 'asset')
        ordering = ['date']
        verbose_name = 'Market Data'
        verbose_name_plural = 'Market Data Records'

    def __str__(self):
        return f"{self.asset_id} - {self.date} - Close: {self.close}"
