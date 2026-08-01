from datetime import timedelta

import numpy as np
import pandas as pd
from django.db.models import F
from django.http import JsonResponse


class PortfolioPnlCalculator:
    def __init__(self, tracking_records, symbols_data, assets):
        self.tracking_records = tracking_records
        self.symbols_data = symbols_data
        self.assets = assets or []

        self.tracking_df = pd.DataFrame(
            list(self.tracking_records.values('date', 'balance'))
        )

        self.initial_record = self.tracking_records.first()
        self.current_record = self.tracking_records.last()

        self.initial_balance = float(self.initial_record.balance)
        self.current_balance = float(self.current_record.balance)
        self.initial_distribution = self.initial_record.distribution_map()
        self.current_distribution = self.current_record.distribution_map()

        self.all_symbols = set(self.initial_distribution.keys()) | set(
            self.current_distribution.keys()
        )

        self.assets_map = {
            asset.get('symbol'): asset
            for asset in self.assets
            if asset.get('symbol')
        }

        self.market_df = self._build_market_df()

    def _build_market_df(self):
        df = pd.DataFrame(list(
            self.symbols_data.values('date', 'close', symbol=F('asset__symbol'))
        ))
        df['date'] = pd.to_datetime(df['date'])
        return df

    def _latest_prices_dict(self):
        latest_prices = self.market_df.loc[
            self.market_df.groupby('symbol')['date'].idxmax()
        ]
        return dict(zip(latest_prices['symbol'], latest_prices['close']))

    def _week_ago_prices_dict(self):
        one_week_ago = pd.Timestamp.now() - timedelta(weeks=1)
        week_ago_data = self.market_df[self.market_df['date'] >= one_week_ago]
        if week_ago_data.empty:
            return {}

        week_ago_prices = week_ago_data.loc[
            week_ago_data.groupby('symbol')['date'].idxmin()
        ]
        return dict(zip(week_ago_prices['symbol'], week_ago_prices['close']))

    def _resolve_quantity(self, symbol, current_asset_value, current_price):
        if symbol in self.assets_map:
            return float(self.assets_map[symbol].get('quantity', 0))
        return current_asset_value / current_price if current_price > 0 else 0

    def _build_pnl_rows(self, latest_prices, week_ago_prices):
        pnl_data = []

        for symbol in self.all_symbols:
            initial_weight = float(self.initial_distribution.get(symbol, 0))
            current_weight = float(self.current_distribution.get(symbol, 0))

            initial_asset_value = self.initial_balance * initial_weight
            current_asset_value = self.current_balance * current_weight

            current_price = float(latest_prices.get(symbol, 0))
            week_ago_price = float(week_ago_prices.get(symbol, current_price))

            if current_price <= 0:
                continue

            quantity = self._resolve_quantity(
                symbol, current_asset_value, current_price
            )

            if quantity <= 0:
                continue

            average_price = initial_asset_value / quantity if quantity > 0 else 0

            pnl_value = current_asset_value - initial_asset_value
            pnl_percent = (
                (current_asset_value - initial_asset_value) / initial_asset_value * 100
            ) if initial_asset_value > 0 else 0

            week_change = (
                (current_price - week_ago_price) / week_ago_price * 100
            ) if week_ago_price > 0 else 0

            pnl_data.append({
                'symbol': symbol,
                'quantity': round(quantity, 2),
                'average_price': round(average_price, 2),
                'current_price': round(current_price, 2),
                'week_ago_price': round(week_ago_price, 2),
                'initial_value': round(initial_asset_value, 2),
                'current_value': round(current_asset_value, 2),
                'pnl_value': round(pnl_value, 2),
                'pnl_percent': round(pnl_percent, 2),
                'week_change_percent': round(week_change, 2),
                'initial_weight': round(initial_weight * 100, 2),
                'current_weight': round(current_weight * 100, 2),
            })

        return pnl_data

    def _balance_volatility(self):
        self.tracking_df['balance'] = pd.to_numeric(
            self.tracking_df['balance'], errors='coerce'
        )
        returns = self.tracking_df['balance'] / self.tracking_df['balance'].shift(1)
        returns = returns.dropna()
        return np.log(returns).std() * np.sqrt(252)

    def calculate(self):
        latest_prices = self._latest_prices_dict()
        week_ago_prices = self._week_ago_prices_dict()
        pnl_data = self._build_pnl_rows(latest_prices, week_ago_prices)

        total_pnl_value = self.current_balance - self.initial_balance
        total_pnl_percent = (
            (self.current_balance - self.initial_balance) / self.initial_balance * 100
        ) if self.initial_balance > 0 else 0

        portfolio_balance_vol = self._balance_volatility()

        return {
            'pnl_data': pnl_data,
            'pnl_general': {
                'initial_balance': round(self.initial_balance, 2),
                'current_balance': round(self.current_balance, 2),
                'total_pnl_value': round(total_pnl_value, 2),
                'total_pnl_percent': round(total_pnl_percent, 2),
                'initial_date': self.initial_record.date.isoformat(),
                'current_date': self.current_record.date.isoformat(),
                'balance_volatility': round(portfolio_balance_vol, 4)
            }
        }
