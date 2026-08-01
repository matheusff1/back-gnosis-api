from django.core.management.base import BaseCommand
from django.db import transaction

from api.market.constants import ALLOWED_SYMBOLS
from api.market.models import Asset, AssetSource, AssetType, DataSource, MarketData
from api.market.seeds import ASSET_SEED, ASSET_TYPES, DATA_SOURCES, FALLBACK_TYPE


class Command(BaseCommand):
    help = 'Semeia o catálogo de ativos (AssetType, DataSource, Asset, AssetSource).'

    @transaction.atomic
    def handle(self, *args, **options):
        allowed = set(ALLOWED_SYMBOLS)

        types = {}
        for slug, description in ASSET_TYPES.items():
            asset_type, _ = AssetType.objects.get_or_create(
                name=slug, defaults={'description': description}
            )
            types[slug] = asset_type

        sources = {}
        for name, description in DATA_SOURCES.items():
            data_source, _ = DataSource.objects.get_or_create(
                name=name, defaults={'description': description}
            )
            sources[name] = data_source

        for symbol, (type_slug, source_name, source_symbol, extra) in ASSET_SEED.items():
            asset, _ = Asset.objects.update_or_create(
                symbol=symbol,
                defaults={
                    'asset_type': types[type_slug],
                    'is_allowed': symbol in allowed,
                },
            )
            AssetSource.objects.update_or_create(
                asset=asset,
                data_source=sources[source_name],
                defaults={
                    'source_symbol': source_symbol,
                    'extra': extra,
                    'is_primary': True,
                },
            )

        # Rede de segurança: qualquer símbolo do banco sem definição no seed.
        db_symbols = set(
            MarketData.objects.values_list('asset__symbol', flat=True).distinct()
        )
        missing = sorted(db_symbols - set(ASSET_SEED.keys()))
        for symbol in missing:
            Asset.objects.get_or_create(
                symbol=symbol,
                defaults={
                    'asset_type': types[FALLBACK_TYPE],
                    'is_allowed': symbol in allowed,
                },
            )

        self.stdout.write(self.style.SUCCESS(
            f'Seed OK: {AssetType.objects.count()} tipos, '
            f'{DataSource.objects.count()} fontes, '
            f'{Asset.objects.count()} ativos '
            f'({Asset.objects.filter(is_allowed=True).count()} allowed), '
            f'{AssetSource.objects.count()} vínculos ativo-fonte.'
        ))
        if missing:
            self.stdout.write(self.style.WARNING(
                f'Símbolos sem definição no seed (tipo "{FALLBACK_TYPE}"): {missing}'
            ))
