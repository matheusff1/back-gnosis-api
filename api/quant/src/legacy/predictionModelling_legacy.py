"""Código legado de modelagem de predições — preservado por organização.

Estas classes/funções foram substituídas pelo pipeline atual em
``api/quant/src/predictionModelling.py`` (``PredictionProcessor`` +
``ModelDataProcesser`` + ``PredictionModelV2``) e NÃO são mais usadas pelo comando
``run_and_save_predictions``. Mantidas aqui apenas para referência histórica.
"""

import traceback

import numpy as np
import pandas as pd
from django.db.models import F
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM
from keras.models import Sequential, clone_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from api.models import MarketData, Prediction
from ..prediction_persistence import persist_prediction
from ..predictionModelling import (
    DEFAULT_ALLOWED_SYMBOLS,
    DEFAULT_CALLBACKS,
    PredictionModelV2,
)


models_cfg = {
    'cp_tanh': {
        'model': 'lstm',
        'window_size': 30,
        'steps_out': 1,
        'pred_seq': 7,
        'epochs': 60,
        'batch_size': 64,
        'scaler_type': 'minmax',
        'train_ratio': 0.80,
        'activation': 'tanh',
        'callbacks': [
            EarlyStopping(monitor='val_mae', baseline=0.05,patience=20, restore_best_weights=True)
        ]
    },
    'af_relu': {
        'model': 'lstm',
        'window_size': 90,
        'steps_out': 7,
        'pred_seq': 7,
        'epochs': 60,
        'batch_size': 32,
        'scaler_type': 'minmax',
        'train_ratio': 0.80,
        'activation': 'relu',
        'callbacks': [
            EarlyStopping(monitor='val_mae', baseline=0.05,patience=20, restore_best_weights=True)
        ]
    },
}

BASE_CFG = models_cfg['cp_tanh']
BASE_WINDOW_SIZES = [BASE_CFG['window_size']]


model_cp_tanh = Sequential([
    LSTM(160, input_shape=(models_cfg['cp_tanh']['window_size'], 1), activation=models_cfg['cp_tanh']['activation'], return_sequences=False),
    Dense(1)
])

model_af_relu = Sequential([
    LSTM(160, input_shape=(models_cfg['af_relu']['window_size'], 1), activation=models_cfg['af_relu']['activation'], return_sequences=False),
    Dense(models_cfg['af_relu']['steps_out'])
])


class PredictionModel:
    def __init__(self, model, callbacks, df, window_size, epochs, batch_size, steps_out, pred_seq, multi_output=False, train_ratio=0.90, scaler_type='minmax'):
        self.df = df
        self.scaler_type = scaler_type
        self.model = clone_model(model)
        self.scaler = scaler_type
        self.window_size = window_size
        self.steps_out = steps_out
        self.pred_seq = pred_seq
        self.train_ratio = train_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.multi_output = multi_output
        self.results = {}

    
    def create_model(self):
        model = Sequential()
        model.add(LSTM(160, input_shape=(self.window_size, 1), activation='tanh', return_sequences=False))
        model.add(Dense(self.steps_out))

        return model
       


    def create_shaped_df(self):
        X, y = [], []

        for i in range(self.window_size, len(self.df) - self.steps_out + 1):
            X.append(self.df.iloc[i - self.window_size:i, :].values)

            y.append(self.df['close'].iloc[i:i + self.steps_out].values)

        return np.array(X), np.array(y)
    
    def split_data(self, X, y):
        train_size = int(len(X) * self.train_ratio)
        
        train_X = X[:train_size]
        train_y = y[:train_size]
        test_X = X[train_size:]
        test_y = y[train_size:]

        print(f'Shapes de train_X: {train_X.shape}, train_y: {train_y.shape}')
        print(f'Shapes de test_X: {test_X.shape}, test_y: {test_y.shape}')

        return train_X, test_X, train_y, test_y

    def scale_data(self, train_X, test_X, train_y, test_y):
        
        if self.scaler_type == 'standard':
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
        elif self.scaler_type == 'minmax':
            scaler_X = MinMaxScaler(feature_range=(0, 1))
            scaler_y = MinMaxScaler(feature_range=(0, 1))
        else:
            raise ValueError("scaler_type deve ser 'standard' ou 'minmax'.")

        shape_X = train_X.shape
        train_X_reshaped = train_X.reshape(-1, shape_X[-1])
        test_X_reshaped = test_X.reshape(-1, shape_X[-1])

        train_X_scaled = scaler_X.fit_transform(train_X_reshaped).reshape(shape_X)
        test_X_scaled = scaler_X.transform(test_X_reshaped).reshape(test_X.shape)

        if self.multi_output: 
            scaler_y.fit(train_y)
            train_y_scaled = scaler_y.transform(train_y)
            test_y_scaled = scaler_y.transform(test_y)
        else:
            train_y = train_y.reshape(-1, 1)
            test_y = test_y.reshape(-1, 1)

            scaler_y.fit(train_y)
            train_y_scaled = scaler_y.transform(train_y).flatten()
            test_y_scaled = scaler_y.transform(test_y).flatten()

        return train_X_scaled, test_X_scaled, train_y_scaled, test_y_scaled, scaler_X, scaler_y
    
    def process_data(self):
        X, y = self.create_shaped_df()
        train_X, test_X, train_y, test_y = self.split_data(X, y)
        train_X_scaled, test_X_scaled, train_y_scaled, test_y_scaled, scaler_X, scaler_y = self.scale_data(train_X, test_X, train_y, test_y)

        return train_X_scaled, test_X_scaled, train_y_scaled, test_y_scaled, scaler_X, scaler_y
    
    def next_days_prediction(self, last_data):
        predictions = []
        current_data = last_data.copy()

        for _ in range(self.pred_seq):
            pred = self.model.predict(current_data)
            predictions.append(pred)
            current_data = np.roll(current_data, -1)
            current_data[-1] = pred  
            
        return np.array(predictions)

    def fit_model(self, train_X, train_y, test_X, test_y):
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        result = self.model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=self.epochs, batch_size=self.batch_size, verbose=0, callbacks=self.callbacks)
        self.results = result

    def predict(self, last_data, scaler_y):
        last_data = last_data.reshape(1, self.window_size, -1)

        print(f'Last data shape: {last_data.shape}')

        if(self.multi_output):
            predictions = self.model.predict(last_data)
            predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1))
        else:
            predictions = self.next_days_prediction(last_data)
            predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1))
        
        print(f'Predictions: {predictions.shape}')

        return predictions
    
    def run(self):
        train_X, test_X, train_y, test_y, scaler_X, scaler_y = self.process_data()
        self.fit_model(train_X, train_y, test_X, test_y)

        last_data = test_X[-1]
        predictions = self.predict(last_data, scaler_y)

        return {
            'predictions': predictions
        }
    

def run_all_predictions(df, model_cfg=models_cfg['cp_tanh'], model=model_cp_tanh):
    predictions = []
    df = df.copy()

    for symbol in df['symbol'].unique():
        print(f'Processing symbol: {symbol}')

        symbol_df = df[df['symbol'] == symbol].reset_index(drop=True)
        symbol_df = symbol_df.sort_values(by='date')
        symbol_df = symbol_df.drop(columns=['symbol', 'date', 'volume', 'high', 'low','open','id'], errors='ignore')
        print(f'Symbol DataFrame shape: {symbol_df.shape}, columns: {symbol_df.columns.tolist()}')
        
        model_runner = PredictionModel(
            model=model,
            callbacks=model_cfg['callbacks'],
            df=symbol_df,
            window_size=model_cfg['window_size'],
            steps_out=model_cfg['steps_out'],
            pred_seq=model_cfg['pred_seq'],
            epochs=model_cfg['epochs'],
            batch_size=model_cfg['batch_size'],
            train_ratio=model_cfg['train_ratio'],
            scaler_type=model_cfg['scaler_type']
        )
        
        result = model_runner.run()

        save = {
            'symbol': symbol,
            'predictions': result['predictions'],
            'results': model_runner.results.history,
            'date': pd.Timestamp.now()
        }

        predictions.append(save)
    
    return predictions

def predictions_process():

    df = pd.DataFrame(list(MarketData.objects.values('date', 'close', 'high', 'low', 'open', 'volume', symbol=F('asset__symbol'))))

    df = df[df['symbol'].isin(DEFAULT_ALLOWED_SYMBOLS)].reset_index(drop=True)

    results = run_all_predictions(df, model_cfg=models_cfg['cp_tanh'], model=model_cp_tanh)
    
    for result in results:
        nested = result.get('results', {})
        persist_prediction({
            'date': pd.Timestamp.now(),
            'symbol': result['symbol'],
            'predictions': result['predictions'],
            'history': nested.get('history'),
            'metrics': nested.get('metrics'),
            'selected_features': nested.get('selected_features'),
            'n_features': nested.get('n_features'),
            'model_config': nested.get('model_config'),
        })


def get_correlation_features(df, target_symbol, allowed_symbols, 
                             pos_threshold=0.70, neg_threshold=-0.45, 
                             max_features=12, min_features=4, 
                             max_inter_corr=0.75):
 
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df deve ser um pandas DataFrame")
    
    required_cols = ['symbol', 'date', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame faltando colunas: {missing_cols}")
    
    if target_symbol not in allowed_symbols:
        raise ValueError(f"target_symbol '{target_symbol}' não está em allowed_symbols")
    
    if target_symbol not in df['symbol'].values:
        raise ValueError(f"Nenhum dado encontrado para symbol '{target_symbol}'")
    
    target_data = df[df['symbol'] == target_symbol]
    if len(target_data) < 2000:
        raise ValueError(f"Dados insuficientes para '{target_symbol}': {len(target_data)} registros. Mínimo: 2000")
    
    try:
        pivot_df = df.pivot_table(
            index='date', 
            columns='symbol', 
            values='close'
        )
    except Exception as e:
        raise ValueError(f"Erro ao criar pivot table: {str(e)}")
    
    available_symbols = [col for col in pivot_df.columns if col in allowed_symbols]
    
    if len(available_symbols) < 2:
        raise ValueError(f"Poucos símbolos disponíveis: {len(available_symbols)}. Mínimo: 2")
    
    symbols_with_enough_data = []
    for sym in available_symbols:
        sym_data = df[df['symbol'] == sym]
        if len(sym_data) >= 2000:
            symbols_with_enough_data.append(sym)
    
    if target_symbol not in symbols_with_enough_data:
        raise ValueError(f"Target '{target_symbol}' não tem 2000+ registros após filtro")
    
    if len(symbols_with_enough_data) < 2:
        raise ValueError(f"Apenas {len(symbols_with_enough_data)} símbolos com 2000+ registros. Mínimo: 2")
    
    pivot_df = pivot_df[symbols_with_enough_data]
    
    pivot_df = pivot_df.ffill().bfill()
    
    pivot_df = pivot_df.dropna(axis=1, how='all')
    
    if len(pivot_df) < 30:
        raise ValueError(f"Poucos dados após tratamento: {len(pivot_df)} linhas. Mínimo: 30")
    
    corr = pivot_df.corr()
    
    if target_symbol not in corr.columns:
        raise ValueError(f"Symbol '{target_symbol}' não tem dados suficientes para correlação")
    
    correlacoes = corr[target_symbol].drop(target_symbol, errors='ignore')
    
    if correlacoes.empty:
        raise ValueError(f"Nenhuma correlação encontrada para '{target_symbol}'")
    
    positivas = correlacoes[correlacoes > pos_threshold].sort_values(ascending=False)
    negativas = correlacoes[correlacoes < neg_threshold].sort_values()
    
    selecionados_pos = [(idx, float(val)) for idx, val in positivas.items()]
    selecionados_neg = [(idx, float(val)) for idx, val in negativas.items()]
    
    selecionados = selecionados_pos + selecionados_neg
    
    if len(selecionados) == 0:
        top_n = correlacoes.abs().sort_values(ascending=False).head(min_features)
        selecionados = [(idx, float(correlacoes[idx])) for idx in top_n.index]
    
    if len(selecionados) > max_features:
        if selecionados_pos and selecionados_neg:
            total = len(selecionados_pos) + len(selecionados_neg)
            quota_pos = max(1, round(max_features * len(selecionados_pos) / total))
            quota_neg = max_features - quota_pos
            selecionados = selecionados_pos[:quota_pos] + selecionados_neg[:quota_neg]
        else:
            top_n = correlacoes.abs().sort_values(ascending=False).head(max_features)
            selecionados = [(idx, float(correlacoes[idx])) for idx in top_n.index]
    
    final_selecionados = []
    
    for nome, val in sorted(selecionados, key=lambda x: abs(x[1]), reverse=True):
        redundante = False
        for f_nome, _ in final_selecionados:
            if nome in corr.index and f_nome in corr.columns:
                inter_corr = abs(corr.loc[nome, f_nome])
                if inter_corr > max_inter_corr:
                    redundante = True
                    break
        
        if not redundante:
            final_selecionados.append((nome, val))
    
    candidatos = correlacoes.drop(
        [n for n, _ in final_selecionados] + [target_symbol], 
        errors='ignore'
    )
    
    while len(final_selecionados) < min_features and not candidatos.empty:
        candidato_nome = candidatos.abs().idxmax()
        candidato_val = float(correlacoes[candidato_nome])
        
        nao_redundante = True
        for f_nome, _ in final_selecionados:
            if candidato_nome in corr.index and f_nome in corr.columns:
                if abs(corr.loc[candidato_nome, f_nome]) > max_inter_corr:
                    nao_redundante = False
                    break
        
        if nao_redundante:
            final_selecionados.append((candidato_nome, candidato_val))
        
        candidatos = candidatos.drop(candidato_nome, errors='ignore')
    
    return {
        'symbol': target_symbol,
        'selected_features': final_selecionados
    }


def prepare_dataframe_for_model(df, target_symbol, selected_features):

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df deve ser um pandas DataFrame")
    
    if 'symbol' not in df.columns or 'date' not in df.columns or 'close' not in df.columns:
        raise ValueError("DataFrame deve conter colunas: symbol, date, close")
    
    if not selected_features:
        raise ValueError("selected_features não pode ser vazio")
    
    try:
        pivot_df = df.pivot_table(
            index='date',
            columns='symbol',
            values='close'
        )
    except Exception as e:
        raise ValueError(f"Erro ao criar pivot table: {str(e)}")
    
    if target_symbol not in pivot_df.columns:
        raise ValueError(f"Target symbol '{target_symbol}' não encontrado nos dados")
    
    feature_symbols = [feat[0] for feat in selected_features]
    selected_cols = [target_symbol] + feature_symbols
    
    valid_features = []
    for feat_sym in feature_symbols:
        feat_data = df[df['symbol'] == feat_sym]
        if len(feat_data) >= 2000 and feat_sym in pivot_df.columns:
            valid_features.append(feat_sym)
    
    selected_cols = [target_symbol] + valid_features
    
    available_cols = [col for col in selected_cols if col in pivot_df.columns]
    
    if target_symbol not in available_cols:
        raise ValueError(f"Target symbol '{target_symbol}' não disponível")
    
    if len(available_cols) < 2:
        raise ValueError(f"Poucas features válidas: {len(available_cols)-1}. Mínimo: 1 feature com 2000+ registros")
    
    model_df = pivot_df[available_cols].copy()
    
    rename_dict = {target_symbol: 'close'}
    for feat_symbol in feature_symbols:
        if feat_symbol in model_df.columns:
            rename_dict[feat_symbol] = f'close_{feat_symbol}'
    
    model_df = model_df.rename(columns=rename_dict)
    
    rows_before = len(model_df)
    model_df = model_df.ffill().bfill()
    
    model_df = model_df.dropna()
    rows_after = len(model_df)
    
    if rows_after == 0:
        raise ValueError("Nenhuma linha restante após tratamento de NaNs")
    
    if rows_after < 100:
        raise ValueError(f"Poucos dados após tratamento: {rows_after} linhas. Mínimo: 100")
    
    model_df = model_df.sort_index()
    
    model_df = model_df.reset_index(drop=True)
    
    return model_df


def run_all_predictions_v2(df, allowed_symbols, callbacks, 
                           window_size=60, epochs=100, batch_size=32, 
                           train_ratio=0.8, scaler_type='minmax',
                           verbose=1, save_data=False,
                           **correlation_params):
   
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df deve ser um pandas DataFrame")
    
    if not allowed_symbols:
        raise ValueError("allowed_symbols não pode ser vazio")
    
    if not isinstance(callbacks, list):
        callbacks = [callbacks] if callbacks else []
    
    predictions = []
    failed_symbols = []
    
    print(f'\n{"="*80}')
    print(f'alloed_symbols: {allowed_symbols}')
    print(f'Iniciando processamento de {len(allowed_symbols)} ativos')
    print(f'{"="*80}\n')
    
    for i, symbol in enumerate(allowed_symbols, 1):
        print(f'\n{"="*80}')
        print(f'[{i}/{len(allowed_symbols)}] Processando: {symbol}')
        print(f'{"="*80}')
        
        try:
            symbol_data = df[df['symbol'] == symbol]
            if len(symbol_data) < 2000:
                error_msg = f"Dados insuficientes: {len(symbol_data)} registros. Mínimo: 2000"
                print(f'Pulando {symbol}: {error_msg}')
                failed_symbols.append({
                    'symbol': symbol,
                    'error': error_msg
                })
                continue
            
            correlation_result = get_correlation_features(
                df=df,
                target_symbol=symbol,
                allowed_symbols=allowed_symbols,
                **correlation_params
            )
            
            selected_features = correlation_result['selected_features']
            
            valid_features = []
            for feat_symbol, corr_val in selected_features:
                feat_data = df[df['symbol'] == feat_symbol]
                if len(feat_data) >= 2000:
                    valid_features.append((feat_symbol, corr_val))
                elif verbose > 0:
                    print(f'   Feature {feat_symbol} descartada: apenas {len(feat_data)} registros')
            
            if len(valid_features) == 0:
                raise ValueError(f"Nenhuma feature válida (com 2000+ registros) encontrada para {symbol}")
            
            if verbose > 0:
                print(f'\nFeatures válidas ({len(valid_features)}):')
                for j, (feat_symbol, corr_val) in enumerate(valid_features, 1):
                    print(f'  {j}. {feat_symbol}: {corr_val:+.4f}')
            
            model_df = prepare_dataframe_for_model(
                df=df,
                target_symbol=symbol,
                selected_features=valid_features  
            )
            
            if verbose > 0:
                print(f'\nDataFrame: shape={model_df.shape}, columns={model_df.columns.tolist()}')
            
            model_runner = PredictionModelV2(
                callbacks=callbacks,
                df=model_df,
                window_size=window_size,
                epochs=epochs,
                batch_size=batch_size,
                train_ratio=train_ratio,
                scaler_type=scaler_type,
                verbose=verbose
            )
            
            result = model_runner.run(save_data=save_data)
            
            model_runner.clear_memory()
            
            save = {
                'symbol': symbol,
                'predictions': result['predictions'],
                'metrics': result['metrics'],
                'history': result['history'].history,
                'selected_features': valid_features,  
                'n_features': len(valid_features) + 1,  
                'date': pd.Timestamp.now(),
                'scaler_X': result['scaler_X'],
                'scaler_y': result['scaler_y'],
                'model_config': {
                    'window_size': window_size,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'scaler_type': scaler_type
                }
            }
            
            predictions.append(save)
            
            if verbose > 0:
                print(f'\n{symbol} concluído!')
                print(f'   Previsões: {result["predictions"]}')
                print(f'   MAE: {result["metrics"]["mae"]:.6f}')
            
        except Exception as e:
            error_msg = f'Erro ao processar {symbol}: {str(e)}'
            print(f'\n{error_msg}')
            
            if verbose > 1:
                traceback.print_exc()
            
            failed_symbols.append({
                'symbol': symbol,
                'error': str(e)
            })
            continue
    
    print(f'\n{"="*80}')
    print(f' Processamento concluído!')
    print(f'   Sucessos: {len(predictions)}/{len(allowed_symbols)}')
    if failed_symbols:
        print(f'   Falhas: {len(failed_symbols)}')
        for fail in failed_symbols:
            print(f'      - {fail["symbol"]}: {fail["error"][:50]}...')
    print(f'{"="*80}\n')
    
    return predictions


def predictions_process_v2(allowed_symbols=DEFAULT_ALLOWED_SYMBOLS, callbacks=DEFAULT_CALLBACKS, 
                          window_size=22, epochs=60, batch_size=32,
                          verbose=1, save_data=False,
                          **kwargs):
  
    print(f'\n{"="*80}')
    print(' Iniciando processo de previsões V2...')
    print(f'{"="*80}\n')
    
    try:
        df = pd.DataFrame(list(MarketData.objects.values('date', 'close', 'high', 'low', 'open', 'volume', symbol=F('asset__symbol'))))
        
        if df.empty:
            raise ValueError("Nenhum dado encontrado em MarketData")
        
        df = df[df['symbol'].isin(allowed_symbols)].reset_index(drop=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date']>='2010-01-01'].reset_index(drop=True)
        
        if df.empty:
            raise ValueError(f"Nenhum dado para os símbolos permitidos: {allowed_symbols}")
        
        print(f'Dados carregados: {len(df)} registros, {df["symbol"].nunique()} símbolos únicos\n')
        
    except Exception as e:
        print(f'Erro ao carregar dados: {str(e)}')
        raise
    
    try:
        results = run_all_predictions_v2(
            df=df,
            allowed_symbols=allowed_symbols,
            callbacks=callbacks,
            window_size=window_size,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            save_data=save_data,
            **kwargs
        )
    except Exception as e:
        print(f' Erro durante processamento: {str(e)}')
        raise
    
    print(f'\n{"="*80}')
    print(' Salvando previsões no banco...')
    print(f'{"="*80}\n')
    
    saved_count = 0
    failed_saves = []
    
    for result in results:
        try:
            persist_prediction(result)
            saved_count += 1

            if verbose > 0:
                print(f'   {result["symbol"]} salvo')
                
        except Exception as e:
            error_msg = f'Erro ao salvar {result["symbol"]}: {str(e)}'
            print(f' {error_msg}')
            failed_saves.append({
                'symbol': result['symbol'],
                'error': str(e)
            })
    
    print(f'\n{"="*80}')
    print(f' Processo concluído!')
    print(f'   Processados: {len(results)} ativos')
    print(f'   Salvos: {saved_count}/{len(results)} previsões')
    if failed_saves:
        print(f'   Falhas ao salvar: {len(failed_saves)}')
        for fail in failed_saves:
            print(f'      - {fail["symbol"]}: {fail["error"][:50]}...')
    print(f'{"="*80}\n')
    
    return results
