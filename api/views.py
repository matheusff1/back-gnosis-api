from datetime import timedelta
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
from django.http import JsonResponse
from .models import MarketData, Prediction
from django.views.decorators.http import require_http_methods
from .src.RiskMeasurements import *
from .src.chat_bot_connection import get_chat_analysis
import json
import traceback
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from .models import Portfolio, PortfolioTracking
from decimal import Decimal, InvalidOperation


ALLOWED_SYMBOLS = [
    'VALE3.SA',
    'PETR4.SA',
    'ITUB4.SA',
    'BBDC4.SA',
    'ABEV3.SA',
    'WEGE3.SA', 
    'B3SA3.SA', 
    'ITSA4.SA', 
    'CSAN3.SA', 
    'BRFS',
    'AAPL',
    'NVDA',
    'MSFT',
    'AMZN',
    'GOOGL',
    'META', 
    'TSLA', 
    'UNH', 
    'DHR', 
    'SPGI']


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

    total_value = Decimal('0')
    asset_values = {}

    try:
        for asset in assets:
            price = Decimal(str(asset.get('price', 0)))
            quantity = Decimal(str(asset.get('quantity', 0)))

            if price <= 0 or quantity <= 0:
                continue  

            value = price * quantity
            asset_values[asset['symbol']] = value
            total_value += value
    except (InvalidOperation, TypeError, KeyError):
        return Response({"error": "Invalid asset format or numeric value."}, status=400)

    if total_value == 0:
        return Response({"error": "Total portfolio value cannot be zero."}, status=400)

    distribution = {
        symbol: float((value / total_value).quantize(Decimal('0.000001')))
        for symbol, value in asset_values.items()
    }

    diff = 1.0 - sum(distribution.values())
    if abs(diff) > 1e-6:
        last_key = list(distribution.keys())[-1]
        distribution[last_key] += diff

    portfolio = Portfolio.objects.create(
        user=user,
        name=name,
        description=description,
        assets=assets,
        initial_distribution=distribution,
        current_distribution=distribution,  
        initial_balance=total_value,
        current_balance=total_value
    )

    PortfolioTracking.objects.create(
        portfolio=portfolio,
        date=pd.Timestamp.now(),
        balance=total_value,
        distribution=distribution
    )

    return Response({
        "id": portfolio.id,
        "name": portfolio.name,
        "description": portfolio.description,
        "initial_balance": float(portfolio.initial_balance),
        "current_balance": float(portfolio.current_balance),
        "initial_distribution": portfolio.initial_distribution,
        "current_distribution": portfolio.current_distribution,
        "assets": portfolio.assets,
        "created_at": portfolio.created_at
    }, status=201)



@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_portfolios(request):
    user = request.user
    portfolios = Portfolio.objects.filter(user=user)

    data = []
    for p in portfolios:
        portfolio_tracking_data = PortfolioTracking.objects.filter(portfolio=p).order_by('date')
        portfolio_initial_balance = portfolio_tracking_data.first().balance if portfolio_tracking_data.exists() else None
        portfolio_current_balance = portfolio_tracking_data.last().balance if portfolio_tracking_data.exists() else None

        portfolio_initial_distribution = portfolio_tracking_data.first().distribution if portfolio_tracking_data.exists() else None
        portfolio_current_distribution = portfolio_tracking_data.last().distribution if portfolio_tracking_data.exists() else None

        data.append({
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "initial_balance": float(portfolio_initial_balance),
            "current_balance": float(portfolio_current_balance),
            "assets": p.assets,
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
        portfolio = Portfolio.objects.filter(user=user, id=portfolio_id).first()
        if not portfolio:
            return JsonResponse({'error': 'Portfolio not found.'}, status=404)

        portfolio_tracking = PortfolioTracking.objects.filter(portfolio=portfolio).order_by('date')

        first_balance = portfolio_tracking.first().balance if portfolio_tracking.exists() else None
        first_distribution = portfolio_tracking.first().distribution if portfolio_tracking.exists() else None
        current_balance = portfolio_tracking.last().balance if portfolio_tracking.exists() else None
        current_distribution = portfolio_tracking.last().distribution if portfolio_tracking.exists() else None
        current_assets = list(current_distribution.keys()) if current_distribution else []

        portfolio_tracking_data = [
            {
                'date': tracking.date.isoformat(),
                'balance': float(tracking.balance),
                'distribution': tracking.distribution
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
        portfolio = Portfolio.objects.filter(user=user, id=portfolio_id).first()
        if not portfolio:
            return JsonResponse({'error': 'Portfolio not found.'}, status=404)
        
        assets = portfolio.assets
        if not assets:
            return JsonResponse({'error': 'Portfolio has no assets.'}, status=400)
        
        symbols = [asset['symbol'] for asset in assets]
        
        symbols_data = MarketData.objects.filter(symbol__in=symbols).order_by('date')
        
        if not symbols_data.exists():
            return JsonResponse({'error': 'No market data found for these symbols.'}, status=404)
        
        df = pd.DataFrame(list(symbols_data.values('symbol', 'date', 'close', 'open', 'high', 'low', 'volume')))
        df['date'] = pd.to_datetime(df['date'])
        
        latest_prices = df.loc[df.groupby('symbol')['date'].idxmax()]
        latest_prices_dict = dict(zip(latest_prices['symbol'], latest_prices['close']))
        
        one_week_ago = pd.Timestamp.now() - timedelta(weeks=1)
        week_ago_data = df[df['date'] >= one_week_ago]
        week_ago_prices = week_ago_data.loc[week_ago_data.groupby('symbol')['date'].idxmin()]
        week_ago_prices_dict = dict(zip(week_ago_prices['symbol'], week_ago_prices['close']))
        
        current_balance = float(portfolio.current_balance) if portfolio.current_balance else float(portfolio.initial_balance)
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
            
            pnl_percent = ((current_asset_value - initial_asset_value) / initial_asset_value * 100) if initial_asset_value > 0 else 0
            
            week_change = ((current_price - week_ago_price) / week_ago_price * 100) if week_ago_price > 0 else 0
            
            total_current_value += Decimal(str(current_asset_value))
            total_initial_value += Decimal(str(initial_asset_value))
            
            pnl_data_temp = {
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
            }
            pnl_data.append(pnl_data_temp)
        
        total_current_value = float(total_current_value)
        total_initial_value = float(total_initial_value)
        
        for item in pnl_data:
            current_weight = (item['current_value'] / total_current_value * 100) if total_current_value > 0 else 0
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
        total_pnl_percent = ((total_current_value - total_initial_value) / total_initial_value * 100) if total_initial_value > 0 else 0
        
        portfolio.current_balance = Decimal(str(total_current_value))
        
        new_current_distribution = {}
        for item in pnl_data:
            new_current_distribution[item['symbol']] = item['current_weight'] / 100
        
        portfolio.current_distribution = new_current_distribution
        portfolio.save()
        
        total_pnl_value = total_current_value - total_initial_value
        total_pnl_percent = ((total_current_value - total_initial_value) / total_initial_value * 100) if total_initial_value > 0 else 0


        return JsonResponse({
            'pnl_data': pnl_data,
            'full_data': {
                'initial_balance': round(total_initial_value, 2),
                'current_balance': round(total_current_value, 2),
                'total_pnl_value': round(total_pnl_value, 2),
                'total_pnl_percent': round(total_pnl_percent, 2),
            }
        }, status=200)
    
    
    
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
        portfolio = Portfolio.objects.filter(user=user, id=portfolio_id).first()
        if not portfolio:
            return JsonResponse({'error': 'Portfolio not found.'}, status=404)
        
        tracking_records = PortfolioTracking.objects.filter(
            portfolio=portfolio
        ).order_by('date')
        
        if not tracking_records.exists():
            return JsonResponse({'error': 'No tracking data found for this portfolio.'}, status=404)
        
        tracking_df = pd.DataFrame(list(tracking_records.values('date', 'balance', 'distribution')))
        
        initial_record = tracking_records.first()
        current_record = tracking_records.last()
        
        initial_balance = float(initial_record.balance)
        current_balance = float(current_record.balance)
        initial_distribution = initial_record.distribution or {}
        current_distribution = current_record.distribution or {}
        
        all_symbols = set(list(initial_distribution.keys()) + list(current_distribution.keys()))
        
        symbols_data = MarketData.objects.filter(symbol__in=all_symbols).order_by('date')
        
        if not symbols_data.exists():
            return JsonResponse({'error': 'No market data found for these symbols.'}, status=404)
        
        df = pd.DataFrame(list(symbols_data.values('symbol', 'date', 'close')))
        df['date'] = pd.to_datetime(df['date'])
        
        latest_prices = df.loc[df.groupby('symbol')['date'].idxmax()]
        latest_prices_dict = dict(zip(latest_prices['symbol'], latest_prices['close']))
        
        one_week_ago = pd.Timestamp.now() - timedelta(weeks=1)
        week_ago_data = df[df['date'] >= one_week_ago]
        week_ago_prices = week_ago_data.loc[week_ago_data.groupby('symbol')['date'].idxmin()]
        week_ago_prices_dict = dict(zip(week_ago_prices['symbol'], week_ago_prices['close']))
        
        assets = portfolio.assets
        assets_map = {asset['symbol']: asset for asset in assets}
        
        pnl_data = []
        
        for symbol in all_symbols:
            initial_weight = float(initial_distribution.get(symbol, 0))
            current_weight = float(current_distribution.get(symbol, 0))
            
            initial_asset_value = initial_balance * initial_weight
            current_asset_value = current_balance * current_weight
            
            current_price = float(latest_prices_dict.get(symbol, 0))
            week_ago_price = float(week_ago_prices_dict.get(symbol, current_price))
            
            if current_price <= 0:
                continue
            
            if symbol in assets_map:
                quantity = float(assets_map[symbol].get('quantity', 0))
            else:
                quantity = current_asset_value / current_price if current_price > 0 else 0
            
            if quantity <= 0:
                continue
            
            average_price = initial_asset_value / quantity if quantity > 0 else 0
            
            pnl_value = current_asset_value - initial_asset_value
            pnl_percent = ((current_asset_value - initial_asset_value) / initial_asset_value * 100) if initial_asset_value > 0 else 0
            
            week_change = ((current_price - week_ago_price) / week_ago_price * 100) if week_ago_price > 0 else 0
            
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
        
        total_pnl_value = current_balance - initial_balance
        total_pnl_percent = ((current_balance - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0
        tracking_df['balance'] = pd.to_numeric(tracking_df['balance'], errors='coerce')
        returns = tracking_df['balance'] / tracking_df['balance'].shift(1)
        returns = returns.dropna()
        portfolio_balance_vol = np.log(returns).std() * np.sqrt(252)


        return JsonResponse({
            'pnl_data': pnl_data,
            'pnl_general': {
                'initial_balance': round(initial_balance, 2),
                'current_balance': round(current_balance, 2),
                'total_pnl_value': round(total_pnl_value, 2),
                'total_pnl_percent': round(total_pnl_percent, 2),
                'initial_date': initial_record.date.isoformat(),
                'current_date': current_record.date.isoformat(),
                'balance_volatility': round(portfolio_balance_vol, 4)
            }
        }, status=200)
    
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def delete_portfolio(request):
    user = request.user
    portfolio_id = request.data.get('id')
    try:
        portfolio = Portfolio.objects.filter(user=user,id=portfolio_id)
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
        portfolio = Portfolio.objects.filter(user=user, id=portfolio_id).first()
        if not portfolio:
            return Response({'error': 'Portfolio not found.'}, status=404)

        assets = [a for a in portfolio.assets if a['symbol'] != symbol]

        total_value = Decimal('0')
        current_distribution = {}
        for asset in assets:
            try:
                price = Decimal(str(asset['price']))
                quantity = Decimal(str(asset['quantity']))
                value = price * quantity
                current_distribution[asset['symbol']] = value
                total_value += value
            except (InvalidOperation, KeyError, TypeError):
                continue  

        if total_value > 0:
            current_distribution = {k: float((v / total_value).quantize(Decimal('0.000001'))) for k, v in current_distribution.items()}
            diff = 1.0 - sum(current_distribution.values())
            if abs(diff) > 1e-6 and current_distribution:
                last_key = list(current_distribution.keys())[-1]
                current_distribution[last_key] += diff
        else:
            current_distribution = {}

        portfolio.assets = assets
        portfolio.current_distribution = current_distribution
        portfolio.current_balance = float(total_value)
        portfolio.save()

        PortfolioTracking.objects.create(
            portfolio=portfolio,
            date=pd.Timestamp.now(),
            balance=portfolio.current_balance,
            distribution=portfolio.current_distribution
        )

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
        portfolio = Portfolio.objects.filter(user=user, id=portfolio_id).first()
        if not portfolio:
            return Response({'error': 'Portfolio not found.'}, status=404)

        assets = portfolio.assets

        if any(a['symbol'] == new_asset['symbol'] for a in assets):
            return Response({'error': 'Asset already exists in the portfolio.'}, status=400)

        assets.append(new_asset)

        total_value = Decimal('0')
        current_distribution = {}
        for asset in assets:
            try:
                price = Decimal(str(asset['price']))
                quantity = Decimal(str(asset['quantity']))
                value = price * quantity
                current_distribution[asset['symbol']] = value
                total_value += value
            except (InvalidOperation, KeyError, TypeError):
                continue

        if total_value > 0:
            current_distribution = {k: float((v / total_value).quantize(Decimal('0.000001'))) for k, v in current_distribution.items()}
            diff = 1.0 - sum(current_distribution.values())
            if abs(diff) > 1e-6 and current_distribution:
                last_key = list(current_distribution.keys())[-1]
                current_distribution[last_key] += diff
        else:
            current_distribution = {}

        portfolio.assets = assets
        portfolio.current_distribution = current_distribution
        portfolio.current_balance = float(total_value)
        portfolio.save()

        PortfolioTracking.objects.create(
            portfolio=portfolio,
            date=pd.Timestamp.now(),
            balance=portfolio.current_balance,
            distribution=portfolio.current_distribution
        )

        return Response({'message': 'Asset added to portfolio successfully.'}, status=200)

    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_portfolio_risk(request):
    user = request.user
    portfolio_id = request.GET.get('id')

    try:
        portfolio = Portfolio.objects.filter(user=user, id=portfolio_id).first()
        if not portfolio:
            return Response({'error': 'Portfolio not found.'}, status=404)
    
        portfolio_current_data = PortfolioTracking.objects.filter(
            portfolio=portfolio
        ).order_by('-date').first()

        assets = portfolio_current_data.distribution.keys()
        print(assets)
        symbols = list(assets)

        distribution_dict = portfolio_current_data.distribution or {}
        distribution = [distribution_dict.get(symbol, 0.0) for symbol in symbols]

        symbols_data = MarketData.objects.filter(symbol__in=symbols).order_by('date')
        symbols_data = pd.DataFrame(list(symbols_data.values('symbol', 'date', 'close', 'high', 'low', 'open', 'volume')))
        if symbols_data.empty:
            return Response({'error': 'No market data found for these symbols.'}, status=404)

        three_years_ago = pd.Timestamp.now() - pd.DateOffset(years=3)
        symbols_data['date'] = pd.to_datetime(symbols_data['date'], errors='coerce')
        symbols_data = symbols_data[symbols_data['date'] >= three_years_ago]

        latest_prices = symbols_data.sort_values('date').drop_duplicates(subset=['symbol'], keep='last')
        price_dict = latest_prices.set_index('symbol')['close'].apply(float).to_dict()

        portfolio_risk = PortfolioRisk(symbols=symbols, distribution=distribution, price_dict=price_dict, df=symbols_data)
        results = portfolio_risk.full_process()

        return Response({
            'symbols': symbols,
            'measures': results,
            'status': 200
        })

    except Exception as e:
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_portfolio_assets_predictions(request):

    user = request.user
    portfolio_id = request.GET.get('id')
    try:
        portfolio = Portfolio.objects.filter(user=user,id=portfolio_id)
        if not portfolio.exists():
            return JsonResponse({'error': 'Portfolio not found.'}, status=404)

        df = pd.DataFrame(list(portfolio.values('assets')))

        assets = df.iloc[0]['assets']
        symbols = [asset['symbol'] for asset in assets]

        predictions = Prediction.objects.filter(symbol__in=symbols).order_by('date')

        predictions_data = pd.DataFrame(list(predictions.values('date', 'symbol', 'prediction')))
        predictions_data = predictions_data.drop_duplicates(subset=['symbol'], keep='last')
        predictions_data['date'] = predictions_data['date'].astype(str)

        predictions_data = json.loads(predictions_data.to_json(orient='records'))
        return JsonResponse({'predictions': predictions_data}, status=200)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_all_symbols(request):
    if request.method == 'GET':
        try:
            symbols = MarketData.objects.order_by('symbol').values_list('symbol', flat=True).distinct()
            symbols = [symb for symb in symbols if symb in ALLOWED_SYMBOLS]
            return JsonResponse({'symbols': list(symbols)}, status=200)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        
    return JsonResponse({'error': 'Invalid request method.'}, status=405)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_asset_historical_data(request, symb):
    if request.method == 'GET':
        symbol = request.GET.get('symbol', symb)
        try:
            market_data = MarketData.objects.filter(symbol=symbol).order_by('date')
            if not market_data.exists():
                return JsonResponse({'error': 'No market data found for the given symbol.'}, status=404)

            df = pd.DataFrame(list(market_data.values('date', 'close','open','high','low','volume')))

            historical_data = df.to_dict(orient='records')
            return JsonResponse({'symbol': symbol, 'historical_data': historical_data}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        
    return JsonResponse({'error': 'Invalid request method.'}, status=405)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_asset_risk_data(request, symb):
    if request.method == 'GET':
        symbol = request.GET.get('symbol', symb)
        try:
            if not symbol or symbol not in ALLOWED_SYMBOLS:
                return JsonResponse({'error': 'Invalid or unsupported symbol.'}, status=400)

            market_data = MarketData.objects.filter(symbol=symbol).order_by('date')

            three_years_ago = pd.Timestamp.now() - pd.DateOffset(years=3)
            market_data = market_data.filter(date__gte=three_years_ago)
            
            if not market_data.exists():
                return JsonResponse({'error': 'No market data found for the given symbol.'}, status=404)

            df = pd.DataFrame(list(market_data.values('date', 'close','high','low','open','volume')))
            df.set_index('date', inplace=True)

            calc = RiskMeasurements(df)
            full_data = calc.full_process()

            import json
            cleaned_data = json.loads(json.dumps(full_data, default=str))
            return JsonResponse({'symbol': symbol, 'full_data': cleaned_data}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=405)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_optimized_portfolio(request):
    if request.method == 'GET':
        data = request.GET
        try:
            user = request.user
            portfolio_id = data.get('portfolio_id')
            min_return = float(data.get('min_return', 0.0006))

            portfolio = Portfolio.objects.filter(user=user,id=portfolio_id)
            portfolio_last_data = PortfolioTracking.objects.filter(
                portfolio=portfolio_id
            ).order_by('-date').first()

            ret = {}

            symbols = portfolio_last_data.distribution.keys()
            symbols = list(symbols)
            symbols_data = MarketData.objects.filter(symbol__in=symbols).order_by('date')

            symbols_data = pd.DataFrame(list(symbols_data.values('symbol', 'date', 'close', 'high', 'low', 'open', 'volume')))

            

            data_mk = process_markowitz_data(symbols_data, behaviour='balanced', min_return=min_return)

            predictions_data = Prediction.objects.filter(symbol__in=symbols).order_by('date')
            predictions_data = pd.DataFrame(list(predictions_data.values('date', 'results', 'symbol', 'prediction')))
            predictions_data = predictions_data.drop_duplicates(subset=['symbol'], keep='last')

            data_gn = process_gnosse_data(symbols_data, predictions_data, behaviour='aggressive')

            try:


                optimization_mk = PortfolioOptimizer(items=data_mk['items'],items_val=data_mk['items_val'], items_returns= data_mk['items_returns'], items_pred= data_mk['items_pred'],
                                                        items_vol= data_mk['items_vol'], min_return=data_mk['min_return'], optimizer= data_mk['optimizer'])
                results_mk = optimization_mk.optimize()

                ret['markowitz'] = results_mk
            except Exception as e:
                traceback.print_exc()
                ret['markowitz'] = {'error': str(e)}

            try:
                optimization_gn = PortfolioOptimizer(items=data_gn['items'],items_val=data_gn['items_val'], items_returns= data_gn['items_returns'], items_pred= data_gn['items_pred'],
                                                        items_vol= data_gn['items_vol'], min_return=data_gn['min_return'], optimizer= data_gn['optimizer'])
                results_gn = optimization_gn.optimize()

                ret['gnosse'] = results_gn

            except Exception as e:
                traceback.print_exc()
                ret['gnosse'] = {'error': str(e)}

            results = {"markowitz": ret['markowitz'], "gnosse": ret['gnosse']}

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
            data = MarketData.objects.all().order_by('date')
            data = pd.DataFrame(list(data.values('symbol', 'date', 'close')))

            data = data[data['symbol'].isin(ALLOWED_SYMBOLS)]
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
            data = data[data['date'] >= pd.Timestamp.now() - pd.DateOffset(years=4)]

            processed_data = process_markowitz_data(data, behaviour='neutral', min_return=0.0005)

            optimizer = PortfolioOptimizer(items=processed_data['items'], items_val=processed_data['items_val'],
                                           items_returns=processed_data['items_returns'], items_pred=processed_data['items_pred'],
                                           items_vol=processed_data['items_vol'], min_return=processed_data['min_return'],
                                           optimizer=processed_data['optimizer'], behaviour=processed_data['behaviour'])
            
            results = optimizer.optimize()
            to_json_results = json.loads(json.dumps(results, default=str))
            return JsonResponse({'symbols': to_json_results['items'], 'distribuition': to_json_results['optimized_weights']  , 'complete_result': to_json_results}, status=200)

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
            market_data = MarketData.objects.filter(symbol=symbol).order_by('date')
            if not market_data.exists():
                return JsonResponse({'error': 'No market data found for the given symbol.'}, status=404)

            df = pd.DataFrame(list(market_data.values('date', 'close','high','low','open','volume')))
            df.set_index('date', inplace=True)

            analysis = get_chat_analysis(symbol, df)

            return JsonResponse({'symbol': symbol, 'chatbot_analysis': analysis}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=405)



@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_assets_last_data(request):
    if request.method == 'GET':
        try:
            symbols = MarketData.objects.order_by('symbol').values_list('symbol', flat=True).distinct()
            symbols = [symb for symb in symbols if symb in ALLOWED_SYMBOLS]

            last_data = {}
            for symbol in symbols:
                asset_data = MarketData.objects.filter(symbol=symbol).order_by('-date').first()
                if asset_data:
                    last_data[symbol] = {
                        'date': asset_data.date,
                        'close': float(asset_data.close),
                        'open': float(asset_data.open),
                        'high': float(asset_data.high),
                        'low': float(asset_data.low),
                        'volume': int(asset_data.volume)
                    }

            return JsonResponse({'last_data': last_data}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        
    return JsonResponse({'error': 'Invalid request method.'}, status=405)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_all_predictions_with_analysis(request):
    if request.method == 'GET':
        try:
            predictions = Prediction.objects.all().order_by('symbol', '-date')
            predictions_df = pd.DataFrame(list(predictions.values('symbol', 'date', 'prediction', 'results')))
            
            predictions_df = predictions_df[predictions_df['symbol'].isin(ALLOWED_SYMBOLS)]
            
            if predictions_df.empty:
                return JsonResponse({'error': 'No predictions found.'}, status=404)

            predictions_df = predictions_df.drop_duplicates(subset=['symbol'], keep='first')
            predictions_df['date'] = predictions_df['date'].astype(str)

            predictions_data = json.loads(predictions_df.to_json(orient='records'))

            symbols_current_data = {}
            for item in predictions_data:
                symbol = item['symbol']
                asset_data = MarketData.objects.filter(symbol=symbol).order_by('-date').first()
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
                prediction = item['prediction']
                current_close = symbols_current_data.get(symbol, {}).get('close')
                
                if current_close and current_close > 0:
                    predicted_return = (prediction - current_close) / current_close
                    predicted_returns[symbol] = predicted_return
                else:
                    predicted_returns[symbol] = None

            processed_predictions_data = []
            for item in predictions_data:
                symbol = item['symbol']
                
                results = item.get('results', {})
                metrics = results.get('metrics', {})
                
                processed_item = {
                    'symbol': symbol,
                    'date': item['date'],
                    'prediction': round(item['prediction'], 2),
                    'current_price': symbols_current_data.get(symbol, {}).get('close'),
                    'predicted_return': round(predicted_returns.get(symbol, 0) * 100, 2) if predicted_returns.get(symbol) else None,
                    
                    'features': results.get('selected_features', []),
                    
                    'metrics': {
                        'mae': round(metrics.get('mae', 0), 4),
                        'mse': round(metrics.get('loss', 0), 4),  
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
            
            mses = [item['metrics']['mse'] for item in processed_predictions_data if item['metrics']['mse'] > 0]
            avg_mse = np.mean(mses) if mses else None

            response = {
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
                    'avg_model_mse': round(avg_mse, 4) if avg_mse else None
                },
                'top_5_best': top_5_best,
                'top_5_worst': top_5_worst,
            }

            return JsonResponse(response, status=200)

        except Exception as e:
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)
        
    return JsonResponse({'error': 'Invalid request method.'}, status=405)



@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_symbols_current_data(request):
    if request.method == 'GET':
        try:
            symbols = MarketData.objects.order_by('symbol').values_list('symbol', flat=True).distinct()
            symbols = [symb for symb in symbols if symb in ALLOWED_SYMBOLS]

            current_data = {}
            for symbol in symbols:
                asset_data = MarketData.objects.filter(symbol=symbol).order_by('-date').first()
                if asset_data:
                    current_data[symbol] = {
                        'date': asset_data.date,
                        'close': float(asset_data.close),
                        'open': float(asset_data.open),
                        'high': float(asset_data.high),
                        'low': float(asset_data.low),
                        'volume': int(asset_data.volume)
                    }

            return JsonResponse({'current_data': current_data}, status=200)

        except Exception as e:

            return JsonResponse({'error': str(e)}, status=500)
        
    return JsonResponse({'error': 'Invalid request method.'}, status=405)