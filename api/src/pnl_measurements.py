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