from django.db import models
from django.conf import settings


class Portfolio(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='portfolios'
    )
    name = models.CharField(max_length=150)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    initial_distribution = models.JSONField(default=dict)

    current_distribution = models.JSONField(default=dict)

    initial_balance = models.DecimalField(max_digits=20, decimal_places=2, default=0)

    current_balance = models.DecimalField(max_digits=20, decimal_places=2, default=0)

    class Meta:
        app_label = 'api'
        db_table = 'portfolio'
        verbose_name = 'Portfolio'
        verbose_name_plural = 'Portfolios'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} - {self.user.email}"

    def holdings(self):
        """Posições atuais da carteira no formato de dicionário (drop-in do
        antigo JSON ``assets``): ``[{symbol, quantity, price, asset_id}]``."""
        return [
            {
                'symbol': holding.asset.symbol,
                'quantity': float(holding.quantity),
                'price': float(holding.price),
                'asset_id': holding.asset_id,
            }
            for holding in self.asset_holdings.select_related('asset')
        ]


class PortfolioTracking(models.Model):
    portfolio = models.ForeignKey(
        Portfolio,
        on_delete=models.CASCADE,
        related_name='tracking_data'
    )
    date = models.DateTimeField()
    balance = models.DecimalField(max_digits=20, decimal_places=2)

    class Meta:
        app_label = 'api'
        db_table = 'portfolio_tracking'
        unique_together = ('portfolio', 'date')
        ordering = ['date']
        verbose_name = 'Portfolio Tracking'
        verbose_name_plural = 'Portfolio Tracking Records'

    def __str__(self):
        return f"{self.portfolio.name} - {self.date} - PnL: {self.balance}"

    def distribution_map(self):
        return {
            position.asset.symbol: float(position.weight)
            for position in self.asset_positions.select_related('asset')
        }


class PortfolioTrackingAsset(models.Model):
    tracking = models.ForeignKey(
        PortfolioTracking,
        on_delete=models.CASCADE,
        related_name='asset_positions'
    )
    asset = models.ForeignKey(
        'api.Asset',
        on_delete=models.PROTECT,
        related_name='tracking_positions'
    )
    weight = models.DecimalField(max_digits=12, decimal_places=8)
    quantity = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    quoted_price = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)

    class Meta:
        app_label = 'api'
        db_table = 'portfolio_tracking_asset'
        unique_together = ('tracking', 'asset')
        verbose_name = 'Portfolio Tracking Asset'
        verbose_name_plural = 'Portfolio Tracking Assets'

    def __str__(self):
        return f"{self.tracking_id} - {self.asset.symbol} - w={self.weight}"


class PortfolioAsset(models.Model):
    portfolio = models.ForeignKey(
        Portfolio,
        on_delete=models.CASCADE,
        related_name='asset_holdings'
    )
    asset = models.ForeignKey(
        'api.Asset',
        on_delete=models.PROTECT,
        related_name='portfolio_holdings'
    )
    quantity = models.DecimalField(max_digits=20, decimal_places=8)
    price = models.DecimalField(max_digits=20, decimal_places=2)

    class Meta:
        app_label = 'api'
        db_table = 'portfolio_asset'
        unique_together = ('portfolio', 'asset')
        verbose_name = 'Portfolio Asset'
        verbose_name_plural = 'Portfolio Assets'

    def __str__(self):
        return f"{self.portfolio_id} - {self.asset.symbol} - qty={self.quantity}"
