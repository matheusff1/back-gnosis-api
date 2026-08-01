"""Views do domínio de mercado (controllers finos)."""

import pandas as pd
from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated

from .models import MarketData
from .services import MarketDataService


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_all_symbols(request):
    if request.method == 'GET':
        try:
            symbols = MarketDataService.available_symbols()
            return JsonResponse({'symbols': symbols}, status=200)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=405)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_asset_historical_data(request, symb):
    if request.method == 'GET':
        symbol = request.GET.get('symbol', symb)
        try:
            market_data = MarketData.objects.filter(asset__symbol=symbol).order_by('date')
            if not market_data.exists():
                return JsonResponse({'error': 'No market data found for the given symbol.'}, status=404)

            df = pd.DataFrame(list(market_data.values('date', 'close', 'open', 'high', 'low', 'volume')))

            historical_data = df.to_dict(orient='records')
            return JsonResponse({'symbol': symbol, 'historical_data': historical_data}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=405)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_assets_last_data(request):
    if request.method == 'GET':
        try:
            last_data = MarketDataService.latest_snapshot_by_symbol()
            return JsonResponse({'last_data': last_data}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=405)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_symbols_current_data(request):
    if request.method == 'GET':
        try:
            current_data = MarketDataService.latest_snapshot_by_symbol()
            return JsonResponse({'current_data': current_data}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=405)
