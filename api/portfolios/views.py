"""Views do domínio de portfólios (controllers finos)."""

import traceback
from decimal import InvalidOperation

from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import Portfolio
from .services import PortfolioService, PortfolioAnalyticsService, PortfolioConfigService
from .src.pnl_measurements import PortfolioPnlCalculator
from ..market.models import MarketData
from ..market.services import MarketDataService


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_portfolio(request):
    user = request.user
    data = request.data

    name = data.get('name')
    description = data.get('description', '')
    assets = data.get('assets', [])

    if not name:
        return Response({"error": "Portfolio name is required."}, status=400)

    if not assets or not isinstance(assets, list):
        return Response({"error": "Assets list is required and must be a list."}, status=400)

    try:
        total_value, distribution = PortfolioService.compute_distribution(
            assets, strict=True, skip_non_positive=True
        )
    except (InvalidOperation, TypeError, KeyError):
        return Response({"error": "Invalid asset format or numeric value."}, status=400)

    if total_value == 0:
        return Response({"error": "Total portfolio value cannot be zero."}, status=400)

    portfolio = PortfolioService.create_portfolio(
        user=user,
        name=name,
        description=description,
        assets=assets,
        distribution=distribution,
        total_value=total_value,
    )

    return Response({
        "id": portfolio.id,
        "name": portfolio.name,
        "description": portfolio.description,
        "initial_balance": float(portfolio.initial_balance),
        "current_balance": float(portfolio.current_balance),
        "initial_distribution": portfolio.initial_distribution,
        "current_distribution": portfolio.current_distribution,
        "assets": portfolio.holdings(),
        "created_at": portfolio.created_at
    }, status=201)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_portfolios(request):
    user = request.user
    portfolios = Portfolio.objects.filter(user=user)

    data = []
    for p in portfolios:
        portfolio_tracking_data = PortfolioService.tracking(p)
        portfolio_initial_balance = portfolio_tracking_data.first().balance if portfolio_tracking_data.exists() else None
        portfolio_current_balance = portfolio_tracking_data.last().balance if portfolio_tracking_data.exists() else None

        portfolio_initial_distribution = portfolio_tracking_data.first().distribution_map() if portfolio_tracking_data.exists() else None
        portfolio_current_distribution = portfolio_tracking_data.last().distribution_map() if portfolio_tracking_data.exists() else None

        data.append({
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "initial_balance": float(portfolio_initial_balance),
            "current_balance": float(portfolio_current_balance),
            "assets": p.holdings(),
            "initial_distribution": portfolio_initial_distribution,
            "current_distribution": portfolio_current_distribution,
            "creation_date": p.created_at.date()
        })

    return Response(data, status=200)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_portfolio(request):
    user = request.user
    portfolio_id = request.GET.get('id')

    if not portfolio_id:
        return JsonResponse({'error': 'Portfolio ID not provided.'}, status=400)

    try:
        portfolio = PortfolioService.get_owned(user, portfolio_id)
        if not portfolio:
            return JsonResponse({'error': 'Portfolio not found.'}, status=404)

        portfolio_tracking = PortfolioService.tracking(portfolio)

        first_balance = portfolio_tracking.first().balance if portfolio_tracking.exists() else None
        first_distribution = portfolio_tracking.first().distribution_map() if portfolio_tracking.exists() else None
        current_balance = portfolio_tracking.last().balance if portfolio_tracking.exists() else None
        current_distribution = portfolio_tracking.last().distribution_map() if portfolio_tracking.exists() else None
        current_assets = list(current_distribution.keys()) if current_distribution else []

        portfolio_tracking_data = [
            {
                'date': tracking.date.isoformat(),
                'balance': float(tracking.balance),
                'distribution': tracking.distribution_map()
            }
            for tracking in portfolio_tracking
        ]

        portfolio_data = {
            "id": portfolio.id,
            "name": portfolio.name,
            "description": portfolio.description,
            "initial_balance": first_balance,
            "current_balance": current_balance,
            "assets": current_assets,
            "initial_distribution": first_distribution,
            "current_distribution": current_distribution,
            "creation_date": portfolio.created_at.date().isoformat(),
            "tracking_data": portfolio_tracking_data
        }

        return JsonResponse({'portfolio': portfolio_data}, status=200)

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def get_and_save_portfolio_pnl(request):
    user = request.user
    portfolio_id = request.data.get('id')

    if not portfolio_id:
        return JsonResponse({'error': 'Portfolio ID not provided.'}, status=400)

    try:
        portfolio = PortfolioService.get_owned(user, portfolio_id)
        if not portfolio:
            return JsonResponse({'error': 'Portfolio not found.'}, status=404)

        if not portfolio.asset_holdings.exists():
            return JsonResponse({'error': 'Portfolio has no assets.'}, status=400)

        result = PortfolioAnalyticsService.save_and_compute_pnl(portfolio)
        if result is None:
            return JsonResponse({'error': 'No market data found for these symbols.'}, status=404)

        return JsonResponse(result, status=200)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_portfolio_pnl(request):
    user = request.user
    portfolio_id = request.GET.get('id')

    if not portfolio_id:
        return JsonResponse({'error': 'Portfolio ID not provided.'}, status=400)

    try:
        portfolio = PortfolioService.get_owned(user, portfolio_id)
        if not portfolio:
            return JsonResponse({'error': 'Portfolio not found.'}, status=404)

        tracking_records = PortfolioService.tracking(portfolio)

        if not tracking_records.exists():
            return JsonResponse({'error': 'No tracking data found for this portfolio.'}, status=404)

        initial_record = tracking_records.first()
        current_record = tracking_records.last()
        initial_distribution = initial_record.distribution_map()
        current_distribution = current_record.distribution_map()
        all_symbols = set(initial_distribution.keys()) | set(current_distribution.keys())

        symbols_data = MarketData.objects.filter(asset__symbol__in=all_symbols).order_by('date')

        if not symbols_data.exists():
            return JsonResponse({'error': 'No market data found for these symbols.'}, status=404)

        calculator = PortfolioPnlCalculator(
            tracking_records=tracking_records,
            symbols_data=symbols_data,
            assets=portfolio.holdings()
        )
        return JsonResponse(calculator.calculate(), status=200)

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def delete_portfolio(request):
    user = request.user
    portfolio_id = request.data.get('id')
    try:
        portfolio = Portfolio.objects.filter(user=user, id=portfolio_id)
        if not portfolio.exists():
            return JsonResponse({'error': 'Portfolio not found.'}, status=404)

        portfolio.delete()
        return JsonResponse({'message': 'Portfolio deleted successfully.'}, status=200)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def delete_asset_from_portfolio(request):
    user = request.user
    portfolio_id = request.data.get('portfolio_id')
    symbol = request.data.get('symbol')

    try:
        portfolio = PortfolioService.get_owned(user, portfolio_id)
        if not portfolio:
            return Response({'error': 'Portfolio not found.'}, status=404)

        assets = [a for a in portfolio.holdings() if a['symbol'] != symbol]
        PortfolioService.apply_assets_update(portfolio, assets)

        return Response({'message': 'Asset removed from portfolio successfully.'}, status=200)

    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def add_asset_to_portfolio(request):
    user = request.user
    portfolio_id = request.data.get('portfolio_id')
    new_asset = request.data.get('asset')

    try:
        portfolio = PortfolioService.get_owned(user, portfolio_id)
        if not portfolio:
            return Response({'error': 'Portfolio not found.'}, status=404)

        assets = portfolio.holdings()

        if any(a['symbol'] == new_asset['symbol'] for a in assets):
            return Response({'error': 'Asset already exists in the portfolio.'}, status=400)

        assets.append(new_asset)
        PortfolioService.apply_assets_update(portfolio, assets)

        return Response({'message': 'Asset added to portfolio successfully.'}, status=200)

    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_portfolio_risk(request):
    user = request.user
    portfolio_id = request.GET.get('id')
    start_date_str = request.GET.get('start_date', None)

    try:
        portfolio = PortfolioService.get_owned(user, portfolio_id)
        if not portfolio:
            return Response({'error': 'Portfolio not found.'}, status=404)

        portfolio_current_data = PortfolioService.latest_tracking(portfolio)

        distribution_dict = portfolio_current_data.distribution_map()
        symbols = list(distribution_dict.keys())
        distribution = [distribution_dict.get(symbol, 0.0) for symbol in symbols]

        symbols_data = MarketDataService.history_df(symbols, start_date=start_date_str)
        if symbols_data.empty:
            return Response({'error': 'No market data found for these symbols.'}, status=404)

        results = PortfolioAnalyticsService.risk_measures(symbols, distribution, symbols_data)
        return Response(results)

    except Exception as e:
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_portfolio_returns_distribution(request):
    user = request.user
    portfolio_id = request.GET.get('id')
    start_date_str = request.GET.get('start_date')

    try:
        portfolio = PortfolioService.get_owned(user, portfolio_id)
        if not portfolio:
            return Response({'error': 'Portfolio not found.'}, status=404)

        tracking_records = PortfolioService.tracking(portfolio)

        if not tracking_records.exists():
            return Response({'error': 'No tracking data found for this portfolio.'}, status=404)

        portfolio_assets_distribution = tracking_records.last().distribution_map()
        assets = list(portfolio_assets_distribution.keys())

        assets_df = MarketDataService.history_df(assets, start_date=start_date_str)
        if assets_df.empty:
            return Response({'error': 'No market data found for the assets in this portfolio.'}, status=404)

        data = PortfolioAnalyticsService.returns_distribution(
            assets, list(portfolio_assets_distribution.values()), assets_df
        )
        return Response(data, status=200)

    except Exception as e:
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_portfolio_accumulated_returns(request):
    user = request.user
    portfolio_id = request.GET.get('id')
    start_date_str = request.GET.get('start_date')

    try:
        portfolio = PortfolioService.get_owned(user, portfolio_id)
        if not portfolio:
            return Response({'error': 'Portfolio not found.'}, status=404)

        tracking_records = PortfolioService.tracking(portfolio)

        if not tracking_records.exists():
            return Response({'error': 'No tracking data found for this portfolio.'}, status=404)

        portfolio_assets_distribution = tracking_records.last().distribution_map()
        assets = list(portfolio_assets_distribution.keys())

        assets_df = MarketDataService.history_df(assets, start_date=start_date_str)
        if assets_df.empty:
            return Response({'error': 'No market data found for assets.'}, status=404)

        data = PortfolioAnalyticsService.accumulated_returns(
            assets, list(portfolio_assets_distribution.values()), assets_df
        )
        return Response(data, status=200)

    except Exception as e:
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_portfolio_config(request):
    user = request.user
    portfolio_id = request.GET.get('id')

    if not portfolio_id:
        return JsonResponse({'error': 'Portfolio ID not provided.'}, status=400)

    portfolio = PortfolioService.get_owned(user, portfolio_id)
    if not portfolio:
        return JsonResponse({'error': 'Portfolio not found.'}, status=404)

    config = PortfolioConfigService.get_or_create(portfolio)
    return JsonResponse({'config': PortfolioConfigService.serialize(config)}, status=200)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_portfolio_config(request):
    user = request.user
    data = request.data
    portfolio_id = data.get('id')

    if not portfolio_id:
        return JsonResponse({'error': 'Portfolio ID not provided.'}, status=400)

    portfolio = PortfolioService.get_owned(user, portfolio_id)
    if not portfolio:
        return JsonResponse({'error': 'Portfolio not found.'}, status=404)

    try:
        config = PortfolioConfigService.update(portfolio, data)
    except ValueError as e:
        return JsonResponse({'error': str(e)}, status=400)
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'config': PortfolioConfigService.serialize(config)}, status=200)
