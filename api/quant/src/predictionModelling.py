from keras.layers import Input
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import traceback
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import clone_model
from django.db.models import F
from api.models import MarketData, Prediction
from .prediction_persistence import persist_prediction


DEFAULT_ALLOWED_SYMBOLS = sorted(
    {
        symbol.strip().upper()
        for symbol in MarketData.objects.values_list('asset__symbol', flat=True)
        if symbol
    }
)





DEFAULT_CALLBACKS = EarlyStopping(
    monitor='mae',
    patience=30,
    baseline=0.05,
    start_from_epoch=30,
    restore_best_weights=True
)

DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_WINDOW_SIZE = 60
DEFAULT_SCALER_TYPE = 'minmax'  
DEFAULT_STEPS_OUT = 5


class PredictionModelV2:
    def __init__(self, callbacks, df, window_size, epochs, batch_size, steps_out, 
                 train_ratio=0.75, scaler_type='minmax', verbose=1):
 
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df deve ser um pandas DataFrame")
        
        if 'close' not in df.columns:
            raise ValueError("DataFrame deve conter coluna 'close'")
        
        if len(df) < window_size + 5:
            raise ValueError(f"DataFrame muito pequeno. Mínimo: {window_size + 5} linhas, atual: {len(df)}")
        
        if df.isnull().any().any():
            raise ValueError("DataFrame contém valores NaN. Limpe os dados antes de usar.")
        
        if scaler_type not in ['minmax', 'standard']:
            raise ValueError("scaler_type deve ser 'minmax' ou 'standard'")
        
        self.df = df.copy()  
        self.scaler_type = scaler_type
        self.window_size = window_size
        self.steps_out = steps_out 
        self.train_ratio = train_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.callbacks = callbacks if callbacks else []
        self.n_features = df.shape[1]
        self.verbose = verbose
        
        self.model = self.create_model()
        
        self.scaler_X = None
        self.scaler_y = None
        
        self.history = None
        
        self.train_X = None
        self.test_X = None
        self.train_y = None
        self.test_y = None
        
        self._is_trained = False

    def create_model(self):
        model = Sequential([
            Input(shape=(self.window_size, self.n_features)),
            LSTM(160, activation='tanh'),
            Dense(self.steps_out)
        ])
        
        if self.verbose > 0:
            print(f'Modelo LSTM criado - Input: ({self.window_size}, {self.n_features}) | Output: {self.steps_out}')
        
        return model

    def create_sequences(self):
        X, y = [], []
        
        for i in range(self.window_size, len(self.df) - self.steps_out + 1):
            X.append(self.df.iloc[i - self.window_size:i, :].values)
            
            y.append(self.df['close'].iloc[i:i + self.steps_out].values)
        
        X = np.array(X)
        y = np.array(y)
        
        if self.verbose > 0:
            print(f'Sequências criadas - X: {X.shape}, y: {y.shape}')
        
        return X, y
    
    def split_data(self, X, y):
        train_size = int(len(X) * self.train_ratio)
        
        train_X = X[:train_size]
        train_y = y[:train_size]
        test_X = X[train_size:]
        test_y = y[train_size:]
        
        if self.verbose > 0:
            print(f'Split - Treino: {train_X.shape[0]} samples | Teste: {test_X.shape[0]} samples')
        
        return train_X, test_X, train_y, test_y
    
    def scale_data(self, train_X, test_X, train_y, test_y):
        if self.scaler_type == 'standard':
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
        else: 
            scaler_X = MinMaxScaler(feature_range=(0, 1))
            scaler_y = MinMaxScaler(feature_range=(0, 1))
        
        original_shape_train = train_X.shape
        original_shape_test = test_X.shape
        
        train_X_2d = train_X.reshape(-1, self.n_features)
        test_X_2d = test_X.reshape(-1, self.n_features)
        
        train_X_scaled = scaler_X.fit_transform(train_X_2d).reshape(original_shape_train)
        test_X_scaled = scaler_X.transform(test_X_2d).reshape(original_shape_test)
        
        train_y_scaled = scaler_y.fit_transform(train_y)
        test_y_scaled = scaler_y.transform(test_y)
        
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        
        if self.verbose > 0:
            print(f'Dados normalizados usando {self.scaler_type}')
        
        return train_X_scaled, test_X_scaled, train_y_scaled, test_y_scaled
    
    def prepare_data(self, save_data=False):
        if self.verbose > 0:
            print('\nPreparando dados...')
        
        X, y = self.create_sequences()
        train_X, test_X, train_y, test_y = self.split_data(X, y)
        train_X_scaled, test_X_scaled, train_y_scaled, test_y_scaled = self.scale_data(
            train_X, test_X, train_y, test_y
        )
        
        if save_data:
            self.train_X = train_X_scaled
            self.test_X = test_X_scaled
            self.train_y = train_y_scaled
            self.test_y = test_y_scaled
        
        return train_X_scaled, test_X_scaled, train_y_scaled, test_y_scaled
    
    def train(self, train_X=None, train_y=None, test_X=None, test_y=None):
        if train_X is None:
            if self.train_X is None:
                raise ValueError("Dados não preparados. Execute prepare_data(save_data=True) primeiro.")
            train_X = self.train_X
            train_y = self.train_y
            test_X = self.test_X
            test_y = self.test_y
        
        if self.verbose > 0:
            print(f'\nTreinando modelo...')
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        try:
            history = self.model.fit(
                train_X, train_y,
                validation_data=(test_X, test_y),
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=0,
                callbacks=self.callbacks
            )
            
            self.history = history
            self._is_trained = True
            
            if self.verbose > 0:
                print(f'Treinamento concluído!')
            
            return history
            
        except Exception as e:
            print(f'Erro durante treinamento: {str(e)}')
            raise
    
    def predict_next_n_days(self, n=5, last_window=None):
        if not self._is_trained:
            raise ValueError("Modelo não treinado. Execute train() primeiro.")
        
        if self.scaler_y is None:
            raise ValueError("Scaler não disponível. Execute prepare_data() primeiro.")
        
        if last_window is None:
            if self.test_X is None:
                raise ValueError("Nenhuma janela fornecida e dados não salvos. Forneça last_window.")
            last_window = self.test_X[-1]
        
        if last_window.ndim == 2:
            last_window = last_window.reshape(1, self.window_size, self.n_features)
        
        try:
            predictions_scaled = self.model.predict(last_window, verbose=0)
            
            predictions = self.scaler_y.inverse_transform(predictions_scaled)
            
            return predictions.flatten()
            
        except Exception as e:
            print(f'Erro durante previsão: {str(e)}')
            raise
    
    def evaluate(self, test_X=None, test_y=None):
        if test_X is None:
            if self.test_X is None or self.test_y is None:
                raise ValueError("Dados de teste não disponíveis.")
            test_X = self.test_X
            test_y = self.test_y
        
        if self.verbose > 0:
            print('\nAvaliando modelo...')
        
        try:
            loss, mae = self.model.evaluate(test_X, test_y, verbose=0)
            
            if self.verbose > 0:
                print(f'Test Loss (MSE): {loss:.6f}')
                print(f'Test MAE: {mae:.6f}')
            
            return {'loss': loss, 'mae': mae}
            
        except Exception as e:
            print(f'Erro durante avaliação: {str(e)}')
            raise
    
    def run(self, save_data=False):
        try:
            train_X, test_X, train_y, test_y = self.prepare_data(save_data=save_data)
            
            self.train(train_X, train_y, test_X, test_y)
            
            metrics = self.evaluate(test_X, test_y)
            
            predictions = self.predict_next_n_days(n=5, last_window=test_X[-1])
            
            if self.verbose > 0:
                print(f'\nPrevisões próximos 5 dias: {predictions}')
            
            return {
                'predictions': predictions,
                'metrics': metrics,
                'history': self.history,
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y
            }
            
        except Exception as e:
            print(f'Erro no pipeline: {str(e)}')
            raise
    
    def get_training_history(self):
        if self.history is None:
            return None
        
        return {
            'loss': self.history.history['loss'],
            'val_loss': self.history.history['val_loss'],
            'mae': self.history.history['mae'],
            'val_mae': self.history.history['val_mae']
        }
    
    def clear_memory(self):
        self.train_X = None
        self.test_X = None
        self.train_y = None
        self.test_y = None
        
        if self.verbose > 0:
            print('Memória limpa')
    




class ModelDataProcesser():
    def __init__(self, full_df, target_symbol, allowed_symbols, 
                 pos_threshold=0.70, neg_threshold=-0.45, 
                 max_features=12, min_features=4, max_inter_corr=0.75, 
                 minimum_samples=2500, years_baseline=10, verbose=1):
        
        self.df = full_df.copy()
        self.target_symbol = target_symbol
        self.allowed_symbols = allowed_symbols
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.max_features = max_features
        self.min_features = min_features
        self.max_inter_corr = max_inter_corr
        self.minimum_samples = minimum_samples
        self.years_baseline = years_baseline
        self.verbose = verbose
        self.start_year = pd.Timestamp.now().year - self.years_baseline

        self._check_data_validity()


    def _check_data_validity(self):
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError("df deve ser um pandas DataFrame.")
        if self.target_symbol not in self.allowed_symbols:
            raise ValueError(f"target_symbol '{self.target_symbol}' não está em allowed_symbols.")

        if self.target_symbol not in self.df['symbol'].values:
            raise ValueError(f"Sem dados para {self.target_symbol} no dataframe.")


    def _filter_df_data(self, df):
        temp_df = df[df['date']>=pd.Timestamp(f'{self.start_year}-01-01')].copy()
        for symbol in self.allowed_symbols:
            symbol_data = temp_df[temp_df['symbol'] == symbol]
            if len(symbol_data) < self.minimum_samples:
                temp_df = temp_df[temp_df['symbol'] != symbol]

        if self.target_symbol not in temp_df['symbol'].values:
            raise ValueError(f"Após filtro, sem dados suficientes para {self.target_symbol}.")
        return temp_df

    def _create_pivot_table(self, df):
        try:
            return df.pivot_table(index='date', columns='symbol', values='close')
        except Exception as e:
            raise ValueError(f"Erro ao pivotear dataframe: {str(e)}")
        

    def _clean_data_table(self, pivot_df):
        pivot_df = pivot_df.ffill().bfill()
        pivot_df = pivot_df.dropna(axis=1, how='all')
        if len(pivot_df) < 30:
            raise ValueError(f"Poucos dados após tratamento: {len(pivot_df)} linhas. Mínimo: 30")
        return pivot_df
    

    def _get_gross_selected_features(self, corr_data):
        correlacoes = corr_data[self.target_symbol].drop(self.target_symbol, errors='ignore')
        
        positivas = correlacoes[correlacoes > self.pos_threshold].sort_values(ascending=False)
        negativas = correlacoes[correlacoes < self.neg_threshold].sort_values()
        
        selecionados_pos = [(idx, float(val)) for idx, val in positivas.items()]
        selecionados_neg = [(idx, float(val)) for idx, val in negativas.items()]
        
        selecionados = selecionados_pos + selecionados_neg
        
        if len(selecionados) == 0:
            top_n = correlacoes.abs().sort_values(ascending=False).head(self.min_features)
            selecionados = [(idx, float(correlacoes[idx])) for idx in top_n.index]
        
        if len(selecionados) > self.max_features:
            if selecionados_pos and selecionados_neg:
                total = len(selecionados_pos) + len(selecionados_neg)
                quota_pos = max(1, round(self.max_features * len(selecionados_pos) / total))
                quota_neg = self.max_features - quota_pos
                selecionados = selecionados_pos[:quota_pos] + selecionados_neg[:quota_neg]
            else:
                top_n = correlacoes.abs().sort_values(ascending=False).head(self.max_features)
                selecionados = [(idx, float(correlacoes[idx])) for idx in top_n.index]
        
        return selecionados
    
    
    def _clean_from_redundant_features(self, gross_selected, corr_data):
        final_selecionados = []
        
        for nome, val in sorted(gross_selected, key=lambda x: abs(x[1]), reverse=True):
            redundante = False
            for f_nome, _ in final_selecionados:
                if nome in corr_data.index and f_nome in corr_data.columns:
                    inter_corr = abs(corr_data.loc[nome, f_nome])
                    if inter_corr > self.max_inter_corr:
                        redundante = True
                        break
            
            if not redundante:
                final_selecionados.append((nome, val))
        
        candidatos = corr_data[self.target_symbol].drop(
            [n for n, _ in final_selecionados] + [self.target_symbol], 
            errors='ignore'
        )
        
        while len(final_selecionados) < self.min_features and not candidatos.empty:
            candidato_nome = candidatos.abs().idxmax()
            candidato_val = float(corr_data[self.target_symbol][candidato_nome])
            
            nao_redundante = True
            for f_nome, _ in final_selecionados:
                if candidato_nome in corr_data.index and f_nome in corr_data.columns:
                    if abs(corr_data.loc[candidato_nome, f_nome]) > self.max_inter_corr:
                        nao_redundante = False
                        break
            
            if nao_redundante:
                final_selecionados.append((candidato_nome, candidato_val))
            
            candidatos = candidatos.drop(candidato_nome, errors='ignore')
        
        return final_selecionados


    def _prepare_dataframe_for_model(self, selected_features):
        try:
            data = self._filter_df_data(self.df)
            pivot_df = self._create_pivot_table(data)
            pivot_df = self._clean_data_table(pivot_df)

            features = [self.target_symbol] + [feat[0] for feat in selected_features]
            model_df = pivot_df[features].copy()
            rename_dict = {self.target_symbol: 'close'}
            for feat_symbol in features:
                if feat_symbol != self.target_symbol:
                    rename_dict[feat_symbol] = f'close_{feat_symbol}'

            model_df.rename(columns=rename_dict, inplace=True)
            return model_df
        except Exception as e:
            raise ValueError(f"Erro ao preparar dataframe para o modelo: {str(e)}")


    def _get_correlation_features(self):
        try:
            cleaned_df = self._filter_df_data(self.df)
            pivot_df = self._create_pivot_table(cleaned_df)
            pivot_df = self._clean_data_table(pivot_df)
            corr_data = pivot_df.corr()
            gross_selected = self._get_gross_selected_features(corr_data)
            final_selected = self._clean_from_redundant_features(gross_selected, corr_data)
            
            return final_selected
        except Exception as e:
            raise ValueError(f"Erro ao obter features de correlação: {str(e)}")
        
    def process(self):
        selected_features = self._get_correlation_features()
        if len(selected_features) == 0:
            raise ValueError("Nenhuma feature selecionada após análise de correlação.")
        
        if self.verbose > 0:
            print(f'\nFeatures válidas ({len(selected_features)}):')

        model_df = self._prepare_dataframe_for_model(selected_features)

        return model_df, selected_features
    







class PredictionProcessor:
    def __init__(self, allowed_symbols=DEFAULT_ALLOWED_SYMBOLS, callbacks=DEFAULT_CALLBACKS, 
                    window_size=DEFAULT_WINDOW_SIZE, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
                    train_ratio=DEFAULT_TRAIN_RATIO, verbose=1, save_data=False, steps_out=DEFAULT_STEPS_OUT
                    ):
            # Data de início do processo. TODAS as predições da rodada são salvas
            # com esta mesma data, mesmo que o treino cruze a meia-noite.
            self.run_date = pd.Timestamp.now()
            self.allowed_symbols = allowed_symbols
            self.callbacks = callbacks
            self.window_size = window_size
            self.epochs = epochs
            self.batch_size = batch_size
            self.verbose = verbose
            self.save_data = save_data
            self.train_ratio = train_ratio
            self.steps_out = steps_out
            self.full_df = self._get_full_dataframe()

    
    def _get_full_dataframe(self):
        all_data = MarketData.objects.values('date', 'close', 'high', 'low', 'open', 'volume', symbol=F('asset__symbol'))
        full_df = pd.DataFrame(list(all_data))
        df = full_df.copy()
        df = df[df['symbol'].isin(self.allowed_symbols)].reset_index(drop=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date']>='2011-01-01'].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"Nenhum dado para os símbolos permitidos: {self.allowed_symbols}")
        return df
    

    def _save_to_db(self, result):
        try:
            persist_prediction(result)
            if self.verbose > 0:
                print(f'   {result["symbol"]} salvo')
            return True
        except Exception as e:
            error_msg = f'Erro ao salvar {result["symbol"]}: {str(e)}'
            print(f' {error_msg}')
            raise

    def predictions_process(self):
        try:
            savemennt_succes = []
            results, errors = self._run_all_predictions()
            for result in results:
                savement_res = self._save_to_db(result)
                if savement_res:
                    savemennt_succes.append(result['symbol'])
            print(f"\n\nProcessamento concluído! {len(savemennt_succes)}/{len(results)} previsões salvas com sucesso.")
        except Exception as e:
            print(f' Erro durante processamento: {str(e)}')
            return
        
    def _run_all_predictions(self):
        predictions = []
        failed_symbols = []

        for i, symbol in enumerate(self.allowed_symbols, 1):
            print(f"\n\n---Processando [{i}/{len(self.allowed_symbols)}]: {symbol} ---")
            try:
                data_processor = ModelDataProcesser(
                    full_df=self.full_df,
                    target_symbol=symbol,
                    allowed_symbols=self.allowed_symbols
                    )
                
                model_df, selected_features = data_processor.process()

                model_runner = PredictionModelV2(
                    callbacks=self.callbacks,
                    df=model_df,
                    window_size=self.window_size,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    train_ratio=self.train_ratio,
                    verbose=self.verbose,
                    steps_out=self.steps_out
                )

                model_result = model_runner.run(save_data=self.save_data)
                model_runner.clear_memory()

                save = {
                    'symbol': symbol,
                    'predictions': model_result['predictions'],
                    'metrics': model_result['metrics'],
                    'history': model_result['history'].history,
                    'selected_features': selected_features,  
                    'n_features': len(selected_features) + 1,  
                    'date': self.run_date,
                    'scaler_X': model_result['scaler_X'],
                    'scaler_y': model_result['scaler_y'],
                    'model_config': {
                        'window_size': self.window_size,
                        'epochs': self.epochs,
                        'batch_size': self.batch_size,
                        'scaler_type': model_runner.scaler_type,
                        'steps_out': model_runner.steps_out
                    }
                }

                predictions.append(save)

                if self.verbose > 0:
                    print(f'\n{symbol} concluído!')
                    print(f'   Previsões: {model_result["predictions"]}')
                    print(f'   MAE: {model_result["metrics"]["mae"]:.6f}')

            except Exception as e:
                error_msg = f"Erro ao processar {symbol}: {str(e)}"
                print(f'\n{error_msg}')
                failed_symbols.append({
                    'symbol': symbol,
                    'error': str(e)
                })
                continue

        return predictions, failed_symbols