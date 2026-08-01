import pandas as pd
from django.db.models import F

from .models import AssetSource, MarketData

OHLCV_COLUMNS = ['symbol', 'date', 'close', 'open', 'high', 'low', 'volume']


class MarketDataService:

    @staticmethod
    def available_symbols():
        return list(
            MarketData.objects
            .filter(asset__is_allowed=True)
            .order_by('asset__symbol')
            .values_list('asset__symbol', flat=True)
            .distinct()
        )

    @staticmethod
    def latest_snapshot_by_symbol(symbols=None):
        if symbols is None:
            symbols = MarketDataService.available_symbols()

        snapshot = {}
        for symbol in symbols:
            asset_data = (
                MarketData.objects.filter(asset__symbol=symbol).order_by('-date').first()
            )
            if asset_data:
                snapshot[symbol] = {
                    'date': asset_data.date,
                    'close': float(asset_data.close),
                    'open': float(asset_data.open),
                    'high': float(asset_data.high),
                    'low': float(asset_data.low),
                    'volume': int(asset_data.volume),
                }
        return snapshot

    @staticmethod
    def history_df(symbols, start_date=None, columns=None, parse_dates=True):
        columns = columns or OHLCV_COLUMNS
        db_columns = [c for c in columns if c != 'symbol']

        queryset = MarketData.objects.filter(asset__symbol__in=symbols)
        if start_date is not None:
            queryset = queryset.filter(date__gte=start_date)
        queryset = queryset.order_by('date')

        values_kwargs = {'symbol': F('asset__symbol')} if 'symbol' in columns else {}
        df = pd.DataFrame(list(queryset.values(*db_columns, **values_kwargs)))
        if parse_dates and not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df


class AssetCatalogService:

    @staticmethod
    def source_symbols(source_name, only_active=True):
        queryset = AssetSource.objects.filter(data_source__name=source_name)
        if only_active:
            queryset = queryset.filter(
                asset__is_active=True, data_source__is_active=True
            )
        return list(queryset.values_list('source_symbol', flat=True))

    @staticmethod
    def bacen_series(only_active=True):
        queryset = AssetSource.objects.filter(data_source__name='bacen')
        if only_active:
            queryset = queryset.filter(
                asset__is_active=True, data_source__is_active=True
            )

        series = {}
        for link in queryset:
            serie = (link.extra or {}).get('sgs_series')
            if serie is not None:
                series[link.source_symbol] = serie
        return series
