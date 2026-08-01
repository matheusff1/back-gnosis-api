"""Persistência relacional das predições.

Concentra a escrita de ``Prediction`` e de suas tabelas relacionais
(``PredictionPoint``, ``PredictionFeature``, ``PredictionTrainingEpoch``) e dos
campos escalares. Usado tanto pelo writer (``predictionModelling``) quanto pelo
backfill, evitando duplicação da lógica de desaninhamento.
"""

from api.models import (
    Asset,
    Prediction,
    PredictionFeature,
    PredictionPoint,
    PredictionTrainingEpoch,
)


def apply_prediction_relations(prediction, *, predictions, history, metrics,
                               selected_features, n_features, model_config):
    metrics = metrics or {}
    config = model_config or {}

    prediction.asset = Asset.objects.filter(symbol=prediction.symbol).first()
    prediction.mae = metrics.get('mae')
    prediction.loss = metrics.get('loss')
    prediction.n_features = n_features
    prediction.window_size = config.get('window_size')
    prediction.epochs = config.get('epochs')
    prediction.batch_size = config.get('batch_size')
    prediction.scaler_type = config.get('scaler_type')
    prediction.steps_out = config.get('steps_out')
    prediction.save(update_fields=[
        'asset', 'mae', 'loss', 'n_features',
        'window_size', 'epochs', 'batch_size', 'scaler_type', 'steps_out',
    ])

    prediction.points.all().delete()
    prediction.features.all().delete()
    prediction.training_epochs.all().delete()

    points = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions or [])
    PredictionPoint.objects.bulk_create([
        PredictionPoint(prediction=prediction, horizon=i + 1, value=float(value))
        for i, value in enumerate(points)
    ])

    selected_features = selected_features or []
    feature_assets = {
        a.symbol: a
        for a in Asset.objects.filter(symbol__in=[f[0] for f in selected_features])
    }
    features = []
    for feature in selected_features:
        asset = feature_assets.get(feature[0])
        if asset is None:
            continue
        features.append(PredictionFeature(
            prediction=prediction, feature_asset=asset, correlation=float(feature[1])
        ))
    PredictionFeature.objects.bulk_create(features)

    history = history or {}
    loss = history.get('loss') or []
    mae = history.get('mae') or []
    val_loss = history.get('val_loss') or []
    val_mae = history.get('val_mae') or []
    n_epochs = max(len(loss), len(mae), len(val_loss), len(val_mae))
    PredictionTrainingEpoch.objects.bulk_create([
        PredictionTrainingEpoch(
            prediction=prediction,
            epoch=i + 1,
            loss=loss[i] if i < len(loss) else None,
            mae=mae[i] if i < len(mae) else None,
            val_loss=val_loss[i] if i < len(val_loss) else None,
            val_mae=val_mae[i] if i < len(val_mae) else None,
        )
        for i in range(n_epochs)
    ])

    return prediction


def persist_prediction(result):
    """Cria uma ``Prediction`` nova e suas relações a partir do ``result`` de
    treino. É o ponto único de escrita usado pelo writer de predições."""
    prediction = Prediction.objects.create(
        date=result['date'],
        symbol=result['symbol'],
    )
    return apply_prediction_relations(
        prediction,
        predictions=result.get('predictions'),
        history=result.get('history'),
        metrics=result.get('metrics'),
        selected_features=result.get('selected_features'),
        n_features=result.get('n_features'),
        model_config=result.get('model_config'),
    )
