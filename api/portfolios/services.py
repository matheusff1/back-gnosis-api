from datetime import timedelta
from decimal import Decimal, InvalidOperation

import pandas as pd

from .constants import WEIGHT_PRECISION
from .models import Portfolio, PortfolioAsset, PortfolioTracking, PortfolioTrackingAsset
from ..market.services import MarketDataService
from ..quant.src.RiskMeasurements import PortfolioRisk
from ..market.models import Asset


class PortfolioService:
    @staticmethod
    def get_owned(user, portfolio_id):
        return Portfolio.objects.filter(user=user, id=portfolio_id).first()

    @staticmethod
    def tracking(portfolio):
        return PortfolioTracking.objects.filter(portfolio=portfolio).order_by('date')

    @staticmethod
    def latest_tracking(portfolio):
        return PortfolioService.tracking(portfolio).last()
    
    @staticmethod
    def _asset_values(assets, strict=False, skip_non_positive=False):
        total_value = Decimal('0')
        values = {}

        for asset in assets:
            if strict:
                price = Decimal(str(asset.get('price', 0)))
                quantity = Decimal(str(asset.get('quantity', 0)))
            else:
                try:
                    price = Decimal(str(asset['price']))
                    quantity = Decimal(str(asset['quantity']))
                except (InvalidOperation, KeyError, TypeError):
                    continue

            if skip_non_positive and (price <= 0 or quantity <= 0):
                continue

            if strict:
                symbol = asset['symbol']
            else:
                try:
                    symbol = asset['symbol']
                except (KeyError, TypeError):
                    continue

            values[symbol] = price * quantity
            total_value += values[symbol]

        return total_value, values

    @staticmethod
    def normalize_weights(values_by_symbol, total_value):
        if total_value <= 0:
            return {}

        distribution = {
            symbol: float((value / total_value).quantize(WEIGHT_PRECISION))
            for symbol, value in values_by_symbol.items()
        }

        diff = 1.0 - sum(distribution.values())
        if abs(diff) > 1e-6 and distribution:
            last_key = list(distribution.keys())[-1]
            distribution[last_key] += diff

        return distribution

    @classmethod
    def compute_distribution(cls, assets, strict=False, skip_non_positive=False):
        total_value, values = cls._asset_values(
            assets, strict=strict, skip_non_positive=skip_non_positive
        )
        return total_value, cls.normalize_weights(values, total_value)

    @staticmethod
    def record_tracking(portfolio):
        tracking = PortfolioTracking.objects.create(
            portfolio=portfolio,
            date=pd.Timestamp.now(),
            balance=portfolio.current_balance,
        )
        PortfolioService._record_tracking_positions(portfolio, tracking)
        return tracking

    @staticmethod
    def _record_tracking_positions(portfolio, tracking):
        weights = portfolio.current_distribution or {}
        positions = [
            PortfolioTrackingAsset(
                tracking=tracking,
                asset=holding.asset,
                weight=Decimal(str(weights.get(holding.asset.symbol, 0))),
                quantity=holding.quantity,
                quoted_price=holding.price,
            )
            for holding in portfolio.asset_holdings.select_related('asset')
        ]
        PortfolioTrackingAsset.objects.bulk_create(positions)

    @staticmethod
    def _replace_holdings(portfolio, assets):        
        by_symbol = {}
        for item in assets:
            symbol = item.get('symbol')
            if symbol:
                by_symbol[symbol] = item

        assets_by_symbol = {
            a.symbol: a
            for a in Asset.objects.filter(symbol__in=list(by_symbol.keys()))
        }

        portfolio.asset_holdings.all().delete()
        holdings = []
        for symbol, item in by_symbol.items():
            asset = assets_by_symbol.get(symbol)
            if asset is None:
                continue
            try:
                quantity = Decimal(str(item.get('quantity', 0)))
                price = Decimal(str(item.get('price', 0)))
            except (InvalidOperation, TypeError):
                continue
            holdings.append(PortfolioAsset(
                portfolio=portfolio, asset=asset, quantity=quantity, price=price
            ))
        PortfolioAsset.objects.bulk_create(holdings)
        return holdings

    @staticmethod
    def create_portfolio(user, name, description, assets, distribution, total_value):
        portfolio = Portfolio.objects.create(
            user=user,
            name=name,
            description=description,
            initial_distribution=distribution,
            current_distribution=distribution,
            initial_balance=total_value,
            current_balance=total_value,
        )
        PortfolioService._replace_holdings(portfolio, assets)
        PortfolioService.record_tracking(portfolio)
        return portfolio

    @staticmethod
    def apply_assets_update(portfolio, assets):
        total_value, distribution = PortfolioService.compute_distribution(assets)

        portfolio.current_distribution = distribution
        portfolio.current_balance = float(total_value)
        portfolio.save()

        PortfolioService._replace_holdings(portfolio, assets)
        PortfolioService.record_tracking(portfolio)
        return portfolio


class PortfolioAnalyticsService:

    @staticmethod
    def risk_measures(symbols, distribution, df):
        latest_prices = df.sort_values('date').drop_duplicates(
            subset=['symbol'], keep='last'
        )
        price_dict = latest_prices.set_index('symbol')['close'].apply(float).to_dict()

        portfolio_risk = PortfolioRisk(
            symbols=symbols, distribution=distribution, price_dict=price_dict, df=df
        )
        return {
            'symbols': symbols,
            'measures': portfolio_risk.full_process(),
            'status': 200,
        }

    @staticmethod
    def returns_distribution(assets, distribution_values, df):
        portfolio_risk = PortfolioRisk(
            symbols=assets, distribution=distribution_values, price_dict={}, df=df
        )

        portfolio_returns = portfolio_risk.portfolio_log_returns()

        individual_returns = portfolio_risk.returns_df
        individual_returns = individual_returns[assets]

        return {
            'returns_columns': assets,
            'returns': individual_returns.values.tolist(),
            'portfolio_returns': portfolio_returns['portfolio_log_returns'],
        }

    @staticmethod
    def accumulated_returns(assets, distribution_values, df):
        portfolio_risk = PortfolioRisk(
            symbols=assets, distribution=distribution_values, price_dict={}, df=df
        )

        portfolio_accum_data = portfolio_risk.portfolio_acumulated_return()
        portfolio_assets_accum_data = (
            portfolio_risk.portfolio_individual_accumulated_return()
        )

        return {
            'accumulated_returns_columns': assets,
            'accumulated_returns': portfolio_assets_accum_data,
            'accumulated_portfolio_returns': portfolio_accum_data,
        }


    ##método inutilizado/legado
    @staticmethod
    def save_and_compute_pnl(portfolio):
        assets = portfolio.holdings()
        symbols = [asset['symbol'] for asset in assets]

        df = MarketDataService.history_df(symbols)
        if df.empty:
            return None

        latest_prices = df.loc[df.groupby('symbol')['date'].idxmax()]
        latest_prices_dict = dict(zip(latest_prices['symbol'], latest_prices['close']))

        one_week_ago = pd.Timestamp.now() - timedelta(weeks=1)
        week_ago_data = df[df['date'] >= one_week_ago]
        week_ago_prices = week_ago_data.loc[
            week_ago_data.groupby('symbol')['date'].idxmin()
        ]
        week_ago_prices_dict = dict(
            zip(week_ago_prices['symbol'], week_ago_prices['close'])
        )

        current_balance = (
            float(portfolio.current_balance)
            if portfolio.current_balance
            else float(portfolio.initial_balance)
        )
        current_distribution = portfolio.current_distribution or {}
        initial_balance = float(portfolio.initial_balance)
        initial_distribution = portfolio.initial_distribution or {}

        pnl_data = []
        total_current_value = Decimal('0')
        total_initial_value = Decimal('0')

        assets_map = {asset['symbol']: asset for asset in assets}

        for symbol in current_distribution.keys():
            initial_weight = float(initial_distribution.get(symbol, 0))

            if symbol not in assets_map:
                continue

            quantity = float(assets_map[symbol].get('quantity', 0))

            if quantity <= 0:
                continue

            current_price = float(latest_prices_dict.get(symbol, 0))
            week_ago_price = float(week_ago_prices_dict.get(symbol, current_price))

            if current_price <= 0:
                continue

            initial_asset_value = initial_balance * initial_weight

            average_price = initial_asset_value / quantity if quantity > 0 else 0

            current_asset_value = quantity * current_price

            pnl_value = current_asset_value - initial_asset_value

            pnl_percent = (
                (current_asset_value - initial_asset_value) / initial_asset_value * 100
            ) if initial_asset_value > 0 else 0

            week_change = (
                (current_price - week_ago_price) / week_ago_price * 100
            ) if week_ago_price > 0 else 0

            total_current_value += Decimal(str(current_asset_value))
            total_initial_value += Decimal(str(initial_asset_value))

            pnl_data.append({
                'symbol': symbol,
                'quantity': quantity,
                'average_price': average_price,
                'current_price': current_price,
                'week_ago_price': week_ago_price,
                'initial_value': initial_asset_value,
                'current_value': current_asset_value,
                'pnl_value': pnl_value,
                'pnl_percent': pnl_percent,
                'week_change_percent': week_change,
                'initial_weight': initial_weight * 100,
            })

        total_current_value = float(total_current_value)
        total_initial_value = float(total_initial_value)

        for item in pnl_data:
            current_weight = (
                item['current_value'] / total_current_value * 100
            ) if total_current_value > 0 else 0
            item['current_weight'] = round(current_weight, 2)

        for item in pnl_data:
            item['quantity'] = round(item['quantity'], 2)
            item['average_price'] = round(item['average_price'], 2)
            item['current_price'] = round(item['current_price'], 2)
            item['week_ago_price'] = round(item['week_ago_price'], 2)
            item['initial_value'] = round(item['initial_value'], 2)
            item['current_value'] = round(item['current_value'], 2)
            item['pnl_value'] = round(item['pnl_value'], 2)
            item['pnl_percent'] = round(item['pnl_percent'], 2)
            item['week_change_percent'] = round(item['week_change_percent'], 2)
            item['initial_weight'] = round(item['initial_weight'], 2)

        total_pnl_value = total_current_value - total_initial_value
        total_pnl_percent = (
            (total_current_value - total_initial_value) / total_initial_value * 100
        ) if total_initial_value > 0 else 0

        portfolio.current_balance = Decimal(str(total_current_value))

        new_current_distribution = {}
        for item in pnl_data:
            new_current_distribution[item['symbol']] = item['current_weight'] / 100

        portfolio.current_distribution = new_current_distribution
        portfolio.save()

        return {
            'pnl_data': pnl_data,
            'full_data': {
                'initial_balance': round(total_initial_value, 2),
                'current_balance': round(total_current_value, 2),
                'total_pnl_value': round(total_pnl_value, 2),
                'total_pnl_percent': round(total_pnl_percent, 2),
            },
        }
