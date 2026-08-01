"""Semeia o catálogo de ativos (idempotente) para clones/deploys futuros.

Popula AssetType, DataSource, Asset e AssetSource a partir das definições
canônicas em ``api.market.seeds`` (mesma fonte usada pelo comando ``seed_assets``).
Em bancos já semeados roda como no-op (``get_or_create``/``update_or_create``).
"""

from django.db import migrations


def seed_catalog(apps, schema_editor):
    from api.market.constants import ALLOWED_SYMBOLS
    from api.market.seeds import ASSET_SEED, ASSET_TYPES, DATA_SOURCES

    AssetType = apps.get_model('api', 'AssetType')
    DataSource = apps.get_model('api', 'DataSource')
    Asset = apps.get_model('api', 'Asset')
    AssetSource = apps.get_model('api', 'AssetSource')

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


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0010_remove_prediction_prediction_and_more'),
    ]

    operations = [
        migrations.RunPython(seed_catalog, migrations.RunPython.noop),
    ]
