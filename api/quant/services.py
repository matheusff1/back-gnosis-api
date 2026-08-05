import json
import traceback

import numpy as np
import pandas as pd
from django.db.models import F

from .constants import (
    ALLOWED_SYMBOLS,
    DEFAULT_MIN_RETURN,
    RISK_LOOKBACK_YEARS,
    STARTER_LOOKBACK_YEARS,
)
from .models import Prediction
from .src.RiskMeasurements import (
    OptmizersDataProcessor,
    PortfolioOptimizer,
    RiskMeasurements,
)
from .src.chat_bot_connection import get_chat_analysis
from ..market.models import MarketData


class AssetRiskService:

    @staticmethod
    def risk_data(symbol):
        market_data = MarketData.objects.filter(asset__symbol=symbol).order_by('date')

        three_years_ago = pd.Timestamp.now() - pd.DateOffset(years=RISK_LOOKBACK_YEARS)
        market_data = market_data.filter(date__gte=three_years_ago)

        if not market_data.exists():
            return None

        df = pd.DataFrame(
            list(market_data.values('date', 'close', 'high', 'low', 'open', 'volume'))
        )
        df.set_index('date', inplace=True)

        full_data = RiskMeasurements(df).full_process()
        return json.loads(json.dumps(full_data, default=str))


class ChatbotAnalysisService:

    @staticmethod
    def analyze(symbol):
        market_data = MarketData.objects.filter(asset__symbol=symbol).order_by('date')
        if not market_data.exists():
            return None

        df = pd.DataFrame(
            list(market_data.values('date', 'close', 'high', 'low', 'open', 'volume'))
        )
        df.set_index('date', inplace=True)

        return get_chat_analysis(symbol, df)


class OptimizationService:

    @staticmethod
    def _optimize_markowitz(data):
        """Otimização Markowitz híbrida, viável para qualquer nº de ativos.

        Tenta os perfis em ordem e retorna o primeiro que resolve:
        1. ``aggressive`` (teto 35%/ativo) — diversificado, mas inviável para
           carteiras muito pequenas (poucos ativos não somam 1 com teto 35%);
        2. ``neutral`` (0–100%) — viável para qualquer N, respeitando ``min_return``;
        3. ``neutral`` sem piso de retorno (min-variância) — último recurso,
           sempre viável, quando nem o retorno mínimo é alcançável.
        """
        attempts = [
            ('aggressive', data['min_return']),
            ('neutral', data['min_return']),
            ('neutral', None),
        ]
        last_error = None
        for behaviour, min_return in attempts:
            try:
                optimizer = PortfolioOptimizer(
                    items=data['items'], items_val=data['items_val'],
                    items_returns=data['items_returns'], items_pred=data['items_pred'],
                    items_vol=data['items_vol'], min_return=min_return,
                    optimizer=data['optimizer'], behaviour=behaviour,
                )
                return optimizer.optimize()
            except Exception as e:
                last_error = e
        raise last_error if last_error else RuntimeError('Otimização Markowitz falhou.')

    @staticmethod
    def optimize_for_symbols(symbols, min_return=DEFAULT_MIN_RETURN):
        symbols_data = MarketData.objects.filter(asset__symbol__in=symbols).order_by('date')
        symbols_data = pd.DataFrame(list(
            symbols_data.values('date', 'close', 'high', 'low', 'open', 'volume',
                                symbol=F('asset__symbol'))
        ))

        data_mk = OptmizersDataProcessor.process_markowitz_data(
            symbols_data, behaviour='aggressive', min_return=min_return
        )

        predictions_data = PredictionService.latest_predictions_dataframe(symbols)

        data_gn = OptmizersDataProcessor.process_gnosse_data(
            symbols_data, predictions_data, behaviour='aggressive'
        )

        ret = {}
        try:
            ret['markowitz'] = OptimizationService._optimize_markowitz(data_mk)
        except Exception as e:
            traceback.print_exc()
            ret['markowitz'] = {'error': str(e)}

        try:
            optimization_gn = PortfolioOptimizer(
                items=data_gn['items'], items_val=data_gn['items_val'],
                items_returns=data_gn['items_returns'], items_pred=data_gn['items_pred'],
                items_vol=data_gn['items_vol'], min_return=data_gn['min_return'],
                optimizer=data_gn['optimizer']
            )
            ret['gnosse'] = optimization_gn.optimize()
        except Exception as e:
            traceback.print_exc()
            ret['gnosse'] = {'error': str(e)}

        return {"markowitz": ret['markowitz'], "gnosse": ret['gnosse']}

    @staticmethod
    def starter_portfolio():
        data = MarketData.objects.filter(asset__is_allowed=True).order_by('date')
        data = pd.DataFrame(list(
            data.values('date', 'close', symbol=F('asset__symbol'))
        ))

        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data = data[
            data['date'] >= pd.Timestamp.now() - pd.DateOffset(years=STARTER_LOOKBACK_YEARS)
        ]

        processed_data = OptmizersDataProcessor.process_markowitz_data(
            data, behaviour='aggressive', min_return=DEFAULT_MIN_RETURN
        )

        results = OptimizationService._optimize_markowitz(processed_data)
        to_json_results = json.loads(json.dumps(results, default=str))
        return {
            'symbols': to_json_results['items'],
            'distribuition': to_json_results['optimized_weights'],
            'complete_result': to_json_results,
        }

    @staticmethod
    def target_weights(symbols, model, horizon=None, min_return=DEFAULT_MIN_RETURN):
        """Pesos-alvo dos símbolos usando um único modelo ('markowitz'|'gnosse').

        Retorna ``{symbol: weight}`` ou ``None`` se não houver dados ou a
        otimização falhar. Para 'gnosse', ``horizon`` seleciona o valor previsto
        (dia N) — casado com a frequência de rebalanceamento da carteira.
        """
        symbols = list(symbols)
        if not symbols:
            return None

        market_df = pd.DataFrame(list(
            MarketData.objects.filter(asset__symbol__in=symbols).order_by('date')
            .values('date', 'close', 'high', 'low', 'open', 'volume', symbol=F('asset__symbol'))
        ))
        if market_df.empty:
            return None

        try:
            if model == 'markowitz':
                data = OptmizersDataProcessor.process_markowitz_data(
                    market_df, behaviour='aggressive', min_return=min_return
                )
                result = OptimizationService._optimize_markowitz(data)
            elif model == 'gnosse':
                predictions_df = PredictionService.latest_predictions_dataframe(symbols)
                if predictions_df.empty:
                    return None
                data = OptmizersDataProcessor.process_gnosse_data(
                    market_df, predictions_df, behaviour='aggressive', horizon=horizon
                )
                optimizer = PortfolioOptimizer(
                    items=data['items'], items_val=data['items_val'],
                    items_returns=data['items_returns'], items_pred=data['items_pred'],
                    items_vol=data['items_vol'], min_return=data['min_return'],
                    optimizer=data['optimizer'],
                )
                result = optimizer.optimize()
            else:
                return None
        except Exception:
            traceback.print_exc()
            return None

        return dict(zip(result['items'], result['optimized_weights']))


class PredictionService:

    @staticmethod
    def _latest_by_symbol(symbols):
        latest = {}
        for prediction in Prediction.objects.filter(
            symbol__in=symbols
        ).order_by('symbol', '-date'):
            if prediction.symbol not in latest:
                latest[prediction.symbol] = prediction
        return latest

    @staticmethod
    def latest_predictions_dataframe(symbols):
        latest = PredictionService._latest_by_symbol(symbols)
        return pd.DataFrame([
            {'symbol': symbol, 'prediction': prediction.prediction_values()}
            for symbol, prediction in latest.items()
        ])

    @staticmethod
    def predictions_for_symbols(symbols):
        latest = PredictionService._latest_by_symbol(symbols)
        return [
            {
                'date': str(prediction.date),
                'symbol': symbol,
                'prediction': prediction.prediction_values(),
            }
            for symbol, prediction in latest.items()
        ]

    @staticmethod
    def all_predictions_with_analysis():
        latest = PredictionService._latest_by_symbol(ALLOWED_SYMBOLS)
        if not latest:
            return None

        predictions_data = []
        for symbol, prediction in latest.items():
            features = [
                [feature.feature_asset.symbol, feature.correlation]
                for feature in prediction.features.select_related('feature_asset')
            ]
            predictions_data.append({
                'symbol': symbol,
                'date': str(prediction.date),
                'prediction': prediction.prediction_values(),
                'results': {
                    'metrics': {'mae': prediction.mae or 0, 'loss': prediction.loss or 0},
                    'selected_features': features,
                },
            })

        symbols_current_data = {}
        for item in predictions_data:
            symbol = item['symbol']
            asset_data = MarketData.objects.filter(asset__symbol=symbol).order_by('-date').first()
            if asset_data:
                symbols_current_data[symbol] = {
                    'date': str(asset_data.date),
                    'close': float(asset_data.close),
                    'open': float(asset_data.open),
                    'high': float(asset_data.high),
                    'low': float(asset_data.low),
                    'volume': int(asset_data.volume)
                }

        predicted_returns = {}
        for item in predictions_data:
            symbol = item['symbol']
            prediction_list = item['prediction']

            if not isinstance(prediction_list, list) or len(prediction_list) == 0:
                predicted_returns[symbol] = None
                continue

            prediction_day5 = prediction_list[-1]

            current_close = symbols_current_data.get(symbol, {}).get('close')

            if current_close and current_close > 0:
                predicted_return = (prediction_day5 - current_close) / current_close
                predicted_returns[symbol] = predicted_return
            else:
                predicted_returns[symbol] = None

        processed_predictions_data = []
        for item in predictions_data:
            symbol = item['symbol']
            prediction_list = item['prediction']

            results = item.get('results', {})
            metrics = results.get('metrics', {})

            if isinstance(prediction_list, list) and len(prediction_list) > 0:
                prediction_day1 = round(prediction_list[0], 2)
                prediction_day5 = round(prediction_list[-1], 2)
                prediction_full = [round(p, 2) for p in prediction_list]
            else:
                prediction_day1 = None
                prediction_day5 = None
                prediction_full = []

            processed_item = {
                'symbol': symbol,
                'date': item['date'],
                'prediction_day1': prediction_day1,
                'prediction_day5': prediction_day5,
                'prediction_full': prediction_full,
                'current_price': symbols_current_data.get(symbol, {}).get('close'),
                'predicted_return': round(predicted_returns.get(symbol, 0) * 100, 2) if predicted_returns.get(symbol) else None,

                'features': results.get('selected_features', []),

                'metrics': {
                    'mae': round(metrics.get('mae', 0), 4),
                    'loss': round(metrics.get('loss', 0), 4),
                }
            }

            processed_predictions_data.append(processed_item)

        valid_returns = [ret for ret in predicted_returns.values() if ret is not None]

        if valid_returns:
            mean_predicted_return = np.mean(valid_returns)
            std_predicted_return = np.std(valid_returns)
            median_predicted_return = np.median(valid_returns)

            highest_return_symbol = max(predicted_returns, key=lambda k: predicted_returns[k] if predicted_returns[k] is not None else float('-inf'))
            lowest_return_symbol = min(predicted_returns, key=lambda k: predicted_returns[k] if predicted_returns[k] is not None else float('inf'))

            positive_predictions = sum(1 for ret in valid_returns if ret > 0)
            negative_predictions = sum(1 for ret in valid_returns if ret <= 0)
        else:
            mean_predicted_return = 0
            std_predicted_return = 0
            median_predicted_return = 0
            highest_return_symbol = None
            lowest_return_symbol = None
            positive_predictions = 0
            negative_predictions = 0

        sorted_returns = sorted(predicted_returns.items(), key=lambda x: x[1] if x[1] is not None else float('-inf'), reverse=True)
        top_5_best = [{'symbol': symbol, 'predicted_return': round(ret * 100, 2)} for symbol, ret in sorted_returns[:5] if ret is not None]
        top_5_worst = [{'symbol': symbol, 'predicted_return': round(ret * 100, 2)} for symbol, ret in sorted_returns[-5:] if ret is not None]

        maes = [item['metrics']['mae'] for item in processed_predictions_data if item['metrics']['mae'] > 0]
        avg_mae = np.mean(maes) if maes else None

        losss = [item['metrics']['loss'] for item in processed_predictions_data if item['metrics']['loss'] > 0]
        avg_loss = np.mean(losss) if losss else None

        return {
            'predictions': processed_predictions_data,
            'summary': {
                'total_assets': len(processed_predictions_data),
                'mean_predicted_return': round(mean_predicted_return * 100, 2),
                'median_predicted_return': round(median_predicted_return * 100, 2),
                'std_predicted_return': round(std_predicted_return * 100, 2),
                'highest_return': {
                    'symbol': highest_return_symbol,
                    'return': round(predicted_returns.get(highest_return_symbol, 0) * 100, 2) if highest_return_symbol else None
                },
                'lowest_return': {
                    'symbol': lowest_return_symbol,
                    'return': round(predicted_returns.get(lowest_return_symbol, 0) * 100, 2) if lowest_return_symbol else None
                },
                'positive_predictions': positive_predictions,
                'negative_predictions': negative_predictions,
                'avg_model_mae': round(avg_mae, 4) if avg_mae else None,
                'avg_model_loss': round(avg_loss, 4) if avg_loss else None
            },
            'top_5_best': top_5_best,
            'top_5_worst': top_5_worst,
            'generated_at': pd.Timestamp.now().isoformat()
        }
