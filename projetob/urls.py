"""
URL configuration for projetob project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from api.views import *
from authapp.views import *



urlpatterns = [
    path('admin/', admin.site.urls),
    path('auth/register/', register_user, name='register'),
    path('auth/login/', login_user, name='login'),
    path('auth/logout/', logout_user, name='logout'),
    path('auth/refresh/', refresh_token, name='token_refresh'),
    path('api/get_all_symbols/', get_all_symbols, name='get_all_symbols'),
    path('api/get_asset_historical_data/<str:symb>/', get_asset_historical_data, name='get_asset_historical_data'),
    path('api/get_asset_risk_data/<str:symb>/', get_asset_risk_data, name='get_asset_risk_data'),
    path('api/get_starter_portfolio/', get_starter_portfolio, name='get_starter_portfolio'),
    path('api/create_portfolio/', create_portfolio, name='create_portfolio'),
    path('api/get_and_save_portfolio_pnl/', get_and_save_portfolio_pnl, name='get_and_save_portfolio_pnl'),
    path('api/get_portfolio_risk/', get_portfolio_risk, name='get_portfolio_risk'),
    path('api/get_optimized_portfolio/', get_optimized_portfolio, name='get_optimized_portfolio'),
    path('api/get_user_portfolios/', get_user_portfolios, name='get_user_portfolios'),
    path('api/get_portfolio_assets_predictions/', get_portfolio_assets_predictions, name='get_portfolio_assets_predictions'),
    path('api/get_assets_last_data/', get_assets_last_data, name='get_assets_last_data'),
    path('api/get_portfolio_pnl/', get_portfolio_pnl, name='get_portfolio_pnl'),
    path('api/get_symbols_current_data/', get_symbols_current_data, name='get_symbols_current_data'),
    path('api/get_portfolio/', get_portfolio, name='get_portfolio'),
    path('api/get_all_predictions/', get_all_predictions_with_analysis, name='get_all_predictions'),
]