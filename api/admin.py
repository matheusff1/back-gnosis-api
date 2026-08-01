from django.contrib import admin

from .models import MarketData
@admin.register(MarketData)
class MarketDataAdmin(admin.ModelAdmin):
    list_display = ('date', 'asset', 'close', 'high', 'low', 'open', 'volume')
    search_fields = ('asset__symbol',)
    list_filter = ('date', 'asset')
    ordering = ('-date',)
    date_hierarchy = 'date'
    
    def has_add_permission(self, request):
        return False
