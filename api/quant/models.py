from django.db import models


class Prediction(models.Model):
    date = models.DateField()
    symbol = models.CharField(max_length=10)
    asset = models.ForeignKey(
        'api.Asset',
        on_delete=models.PROTECT,
        related_name='predictions',
        null=True,
        blank=True,
    )

    # Métricas finais do modelo (antes em results['metrics']).
    mae = models.FloatField(null=True, blank=True)
    loss = models.FloatField(null=True, blank=True)

    # Configuração do modelo (antes em results['model_config'] / n_features).
    n_features = models.IntegerField(null=True, blank=True)
    window_size = models.IntegerField(null=True, blank=True)
    epochs = models.IntegerField(null=True, blank=True)
    batch_size = models.IntegerField(null=True, blank=True)
    scaler_type = models.CharField(max_length=50, null=True, blank=True)
    steps_out = models.IntegerField(null=True, blank=True)
    class Meta:
        app_label = 'api'
        db_table = 'predictions'
        unique_together = ('date', 'symbol')
        ordering = ['date']
        verbose_name = 'Prediction'
        verbose_name_plural = 'Predictions'

    def __str__(self):
        return f"{self.symbol} - {self.date}"

    def prediction_values(self):
        return [float(point.value) for point in self.points.order_by('horizon')]


class PredictionPoint(models.Model):
    prediction = models.ForeignKey(
        Prediction, on_delete=models.CASCADE, related_name='points'
    )
    horizon = models.IntegerField()
    value = models.FloatField()

    class Meta:
        app_label = 'api'
        db_table = 'prediction_point'
        unique_together = ('prediction', 'horizon')
        ordering = ['horizon']
        verbose_name = 'Prediction Point'
        verbose_name_plural = 'Prediction Points'

    def __str__(self):
        return f"{self.prediction_id} - d{self.horizon} = {self.value}"


class PredictionFeature(models.Model):
    prediction = models.ForeignKey(
        Prediction, on_delete=models.CASCADE, related_name='features'
    )
    feature_asset = models.ForeignKey(
        'api.Asset', on_delete=models.PROTECT, related_name='feature_in_predictions'
    )
    correlation = models.FloatField()

    class Meta:
        app_label = 'api'
        db_table = 'prediction_feature'
        unique_together = ('prediction', 'feature_asset')
        verbose_name = 'Prediction Feature'
        verbose_name_plural = 'Prediction Features'

    def __str__(self):
        return f"{self.prediction_id} - {self.feature_asset_id} ({self.correlation})"


class PredictionTrainingEpoch(models.Model):
    prediction = models.ForeignKey(
        Prediction, on_delete=models.CASCADE, related_name='training_epochs'
    )
    epoch = models.IntegerField()
    loss = models.FloatField(null=True, blank=True)
    mae = models.FloatField(null=True, blank=True)
    val_loss = models.FloatField(null=True, blank=True)
    val_mae = models.FloatField(null=True, blank=True)

    class Meta:
        app_label = 'api'
        db_table = 'prediction_training_epoch'
        unique_together = ('prediction', 'epoch')
        ordering = ['epoch']
        verbose_name = 'Prediction Training Epoch'
        verbose_name_plural = 'Prediction Training Epochs'

    def __str__(self):
        return f"{self.prediction_id} - epoch {self.epoch}"
