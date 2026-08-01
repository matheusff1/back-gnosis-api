"""Views do domínio quant/modelling (controllers finos)."""

import traceback

from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated

from .constants import ALLOWED_SYMBOLS, DEFAULT_MIN_RETURN
from .services import (
    AssetRiskService,
    ChatbotAnalysisService,
    OptimizationService,
    PredictionService,
)
from ..portfolios.services import PortfolioService


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_asset_risk_data(request, symb):
    if request.method == 'GET':
        symbol = request.GET.get('symbol', symb)
        try:
            if not symbol or symbol not in ALLOWED_SYMBOLS:
                return JsonResponse({'error': 'Invalid or unsupported symbol.'}, status=400)

            full_data = AssetRiskService.risk_data(symbol)
            if full_data is None:
                return JsonResponse({'error': 'No market data found for the given symbol.'}, status=404)

            return JsonResponse({'symbol': symbol, 'full_data': full_data}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=405)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_optimized_portfolio(request):
    if request.method == 'GET':
        data = request.GET
        try:
            portfolio_id = data.get('portfolio_id')
            min_return = float(data.get('min_return', DEFAULT_MIN_RETURN))

            portfolio_last_data = PortfolioService.latest_tracking(portfolio_id)
            symbols = list(portfolio_last_data.distribution_map().keys())

            results = OptimizationService.optimize_for_symbols(symbols, min_return)

            return JsonResponse({'results': results}, status=200)

        except Exception as e:
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=405)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_starter_portfolio(request):
    if request.method == 'GET':
        try:
            results = OptimizationService.starter_portfolio()
            return JsonResponse(results, status=200)

        except Exception as e:
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=405)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_portfolio_assets_predictions(request):
    user = request.user
    portfolio_id = request.GET.get('id')
    try:
        portfolio = PortfolioService.get_owned(user, portfolio_id)
        if not portfolio:
            return JsonResponse({'error': 'Portfolio not found.'}, status=404)

        symbols = [holding['symbol'] for holding in portfolio.holdings()]

        predictions_data = PredictionService.predictions_for_symbols(symbols)
        return JsonResponse({'predictions': predictions_data}, status=200)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_all_predictions_with_analysis(request):
    if request.method == 'GET':
        try:
            response = PredictionService.all_predictions_with_analysis()
            if response is None:
                return JsonResponse({'error': 'No predictions found.'}, status=404)

            return JsonResponse(response, status=200)

        except Exception as e:
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=405)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_asset_chatbot_analysis(request, symb):
    if request.method == 'GET':
        symbol = request.GET.get('symbol', symb)
        try:
            analysis = ChatbotAnalysisService.analyze(symbol)
            if analysis is None:
                return JsonResponse({'error': 'No market data found for the given symbol.'}, status=404)

            return JsonResponse({'symbol': symbol, 'chatbot_analysis': analysis}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=405)
