import pandas as pd
import numpy as np
import cvxpy as cp
import json

class RiskMeasurements:
    def __init__(self, df):
        self.df = df.copy()
        self.df.dropna(inplace=True)

        if self.df.index.name == 'date':
            self.df.reset_index(inplace=True)

        self.df = self.df.sort_values(by='date')
        self.df['close'] = pd.to_numeric(self.df['close'], errors='coerce')
        self.df['high'] = pd.to_numeric(self.df['high'], errors='coerce')
        self.df['low'] = pd.to_numeric(self.df['low'], errors='coerce')
        self.df.dropna(subset=['close', 'high', 'low'], inplace=True)
        self.__log_returns()

    def __log_returns(self):
        self.df['Log_Returns'] = np.log(self.df['close'] / self.df['close'].shift(1))

    def log_returns(self):
        return self.df[['date', 'Log_Returns']].dropna()

    def historical_volatility(self):
        log_ret = self.df['Log_Returns'].dropna()
        std_log_ret = log_ret.std()
        return {
            'vol_per_year': std_log_ret * np.sqrt(252),
            'vol_per_month': std_log_ret * np.sqrt(21),
            'vol_per_week': std_log_ret * np.sqrt(5),
            'vol_per_day': std_log_ret
        }

    def parametric_var(self, z=1.96, confidence_level=0.95):
        vols = self.historical_volatility()
        last_price = self.df['close'].iloc[-1]
        return {
            'value_at_risk': -z * vols['vol_per_day'] * last_price,
            'value_at_risk_annualized': -z * vols['vol_per_year'] * last_price,
            'value_at_risk_monthly': -z * vols['vol_per_month'] * last_price,
            'value_at_risk_weekly': -z * vols['vol_per_week'] * last_price,
            'confidence_level': confidence_level
        }
    
    def daily_amplitude(self):
        self.df['amplitude'] = self.df['high'] - self.df['low']
        return self.df[['date', 'amplitude']].dropna()

    def sharpe_ratio(self, risk_free_rate=0.06):
        log_ret = self.df['Log_Returns'].dropna()
        mean_ret = log_ret.mean() * 252
        vol = self.historical_volatility()['vol_per_year']
        return {
            'sharpe_ratio': (mean_ret - risk_free_rate) / vol,
            'risk_free_rate': risk_free_rate
        }

    def max_drawdown(self):
        log_ret = self.df['Log_Returns'].dropna()
        cumulative = np.exp(log_ret.cumsum())
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_date': self.df['date'].iloc[drawdown.idxmin()],
            'max_drawdown_value': cumulative.min()
        }

    def kurtosis(self):
        return {
            'kurtosis': self.df['Log_Returns'].dropna().kurtosis()
        }

    def full_process(self):
        return {
            'historical_volatility': self.historical_volatility(),
            'parametric_var': self.parametric_var(),
            'sharpe_ratio': self.sharpe_ratio(),
            'max_drawdown': self.max_drawdown(),
            'kurtosis': self.kurtosis(),
            'daily_amplitude': self.daily_amplitude(),
        }


class PortfolioOptimizer:
    def __init__(self, items, items_val, items_returns=None, items_pred=None, items_vol=None, min_return=None, optimizer='gnosse', behaviour='conservative'):
        self.items = items
        self.items_val = items_val
        self.items_pred = items_pred
        self.items_vol = items_vol
        self.items_returns = items_returns
        self.min_return = min_return
        self.optimizer = optimizer
        self.behaviour = behaviour

    def __define_quantiles(self):
        cv = self.items_vol.std() / self.items_vol.mean()
        if cv < 0.10:
            return (5, 95)
        elif cv < 0.20:
            return (10, 90)
        elif cv < 0.35:
            return (20, 80)
        else:
            return (25, 50, 75)

    def __calc_bounds(self, optimizer, behaviour='conservative', percentils=None):
        bounds = []

        if optimizer == 'gnosse':
            quantiles = np.percentile(self.items_vol, percentils)
            if len(quantiles) == 2:
                q1, q2 = quantiles
            elif len(quantiles) == 3:
                q1, q2, q3 = quantiles

            for vol in self.items_vol:
                if behaviour == 'conservative':
                    if len(quantiles) == 2:
                        if vol < q1:
                            bounds.append((0.1, 0.35))
                        elif vol < q2:
                            bounds.append((0.05, 0.25))
                        else:
                            bounds.append((0.0, 0.15))
                    else:
                        if vol < q1:
                            bounds.append((0.1, 0.25))
                        elif vol < q2:
                            bounds.append((0.05, 0.20))
                        elif vol < q3:
                            bounds.append((0.025, 0.15))
                        else:
                            bounds.append((0.0, 0.10))
                elif behaviour == 'moderate':
                    if len(quantiles) == 2:
                        if vol < q1:
                            bounds.append((0.05, 0.30))
                        elif vol < q2:
                            bounds.append((0.025, 0.25))
                        else:
                            bounds.append((0.0, 0.25))
                    else:
                        if vol < q1:
                            bounds.append((0.05, 0.30))
                        elif vol < q2:
                            bounds.append((0.025, 0.25))
                        elif vol < q3:
                            bounds.append((0.020, 0.25))
                        else:
                            bounds.append((0.0, 0.20))
                elif behaviour == 'aggressive':
                    if len(quantiles) == 2:
                        if vol < q1:
                            bounds.append((0.0, 0.35))
                        elif vol < q2:
                            bounds.append((0.0, 0.40))
                        else:
                            bounds.append((0.0, 0.50))
                    else:
                        if vol < q1:
                            bounds.append((0.0, 0.30))
                        elif vol < q2:
                            bounds.append((0.0, 0.40))
                        elif vol < q3:
                            bounds.append((0.0, 0.50))
                        else:
                            bounds.append((0.0, 0.60))
                else:
                    raise ValueError("Behaviour type must be 'conservative', 'moderate' or 'aggressive'.")

        elif optimizer in ['gnosse2', 'markowitz']:
            for _ in self.items:
                if behaviour == 'conservative':
                    bounds.append((0.1, 0.5))
                elif behaviour == 'moderate':
                    bounds.append((0.05, 0.4))
                elif behaviour == 'aggressive':
                    bounds.append((0.0, 0.35))
                elif behaviour == 'neutral' and optimizer == 'markowitz':
                    bounds.append((0.0, 1.0))
                else:
                    raise ValueError("Invalid behaviour for optimizer")
        else:
            raise ValueError("Invalid optimizer type")

        return bounds

    def __returns_means(self):
        returns_means = np.zeros(len(self.items))
        for i, item in enumerate(self.items):
            if item in self.items_returns:
                returns_means[i] = self.items_returns[item].mean()
        return returns_means

    def __define_coefficients(self):
        if self.optimizer == 'gnosse':
            return (self.items_pred - self.items_val) / self.items_val
        elif self.optimizer == 'gnosse2':
            return ((self.items_pred - self.items_val) / self.items_val) / self.items_vol
        else:
            return None

    def optimize(self):
        n = len(self.items)
        bounds = self.__calc_bounds(self.optimizer, self.behaviour, self.__define_quantiles())
        x = cp.Variable(n)
        constraints = [cp.sum(x) == 1]

        for i in range(n):
            constraints.append(x[i] >= bounds[i][0])
            constraints.append(x[i] <= bounds[i][1])

        if self.optimizer == 'markowitz':
            cov_matrix = self.items_returns.cov().values
            risk = cp.quad_form(x, cov_matrix)
            objective = cp.Minimize(risk)
            if self.min_return is not None:
                returns = self.__returns_means()
                constraints.append(returns @ x >= self.min_return)

        else:
            coefficients = self.__define_coefficients()
            expected_gain = coefficients @ x
            objective = cp.Maximize(expected_gain)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, verbose=False)

        if x.value is None:
            raise ValueError("Otimização falhou.")

        return {
            'optimized_weights': x.value.tolist(),
            'items': self.items,
            'optimizer': self.optimizer,
        }
    
class PortfolioRisk:
    def __init__(self, symbols, distribution, price_dict, df):
        self.symbols = symbols
        self.distribution = np.array(distribution, dtype=float)
        self.price_dict = price_dict
        self.df = df.copy()
        self.returns_df = self.__gen_log_returns_df()

    def __gen_adjusted_dates_df(self):
        total_symbol = self.df['symbol'].nunique()
        symbols_per_date = self.df.groupby('date')['symbol'].nunique()

        common_dates = symbols_per_date[symbols_per_date == total_symbol].index

        df_common_dates = self.df[self.df['date'].isin(common_dates)].sort_values(by='date')

        return df_common_dates

    def __gen_log_returns_df(self):
        df_common_dates = self.__gen_adjusted_dates_df()
        log_returns = {}

        for symbol in self.symbols:
            df_symbol = df_common_dates[df_common_dates['symbol'] == symbol].copy()
            
            df_symbol['close'] = pd.to_numeric(df_symbol['close'], errors='coerce')
            df_symbol = df_symbol.dropna(subset=['close'])

            log_return = np.log(df_symbol['close'] / df_symbol['close'].shift(1))
            log_return = log_return.dropna().reset_index(drop=True)

            log_returns[symbol] = log_return


        log_returns_df = pd.DataFrame(log_returns)
        return log_returns_df

   
    def __weights(self):
        prices = np.array([self.price_dict[s] for s in self.symbols])
        nominal_values = prices * self.distribution
        total_value = nominal_values.sum()
        return nominal_values / total_value

    def portfolio_volatility(self):
        weights = self.__weights()
        cov_matrix = self.returns_df.cov()
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
        return {
            'portfolio_volatility': portfolio_vol,
            'portfolio_volatility_annualized': portfolio_vol * np.sqrt(252),
            'portfolio_volatility_monthly': portfolio_vol * np.sqrt(21),
            'portfolio_volatility_weekly': portfolio_vol * np.sqrt(5),
        }

    def portfolio_correlation(self):
        corr_matrix = self.returns_df[self.symbols].corr()
        return {
            'correlation_dict': corr_matrix.to_dict()
        }

    def individual_volatilities(self):
        vols = {}
        for symbol in self.symbols:
            risk = RiskMeasurements(self.df[self.df['symbol'] == symbol])
            vol = risk.historical_volatility()
            vols[symbol] = vol
        return {
            'individual_volatilities': vols,
        }

    def portfolio_var(self, z=1.65):
        vol_data = self.portfolio_volatility()
        portfolio_vol = vol_data['portfolio_volatility']
        total_value = np.sum([self.price_dict[s] * w for s, w in zip(self.symbols, self.distribution)])
        var = z * portfolio_vol * total_value
        return {
            'portfolio_value': total_value,
            'value_at_risk': -var,
            'var_annualized': -var * np.sqrt(252),
            'var_monthly': -var * np.sqrt(21),
            'var_weekly': -var * np.sqrt(5),
            'confidence_level': '95%' 
        }

    def full_process(self):
        return {
            'portfolio_volatility': self.portfolio_volatility(),
            'individual_volatilities': self.individual_volatilities(),
            'portfolio_var': self.portfolio_var(),
            'portfolio_correlation': self.portfolio_correlation(),
            'weights': self.__weights().tolist(),
        }


    
def process_markowitz_data(df, behaviour='conservative',min_return=0.0007):
    df = df.copy()
    total_symbol = df['symbol'].nunique()
    symbols_per_date = df.groupby('date')['symbol'].nunique()

    common_dates = symbols_per_date[symbols_per_date == total_symbol].index

    df_common_dates = df[df['date'].isin(common_dates)]

    df_common_dates['close'] = df_common_dates['close'].astype(float)

    pivot = df_common_dates.pivot(index='date', columns='symbol', values='close')

    last_values = pivot.loc[pivot.index.max()]

    returns = np.log(pivot / pivot.shift(1)).dropna()

    symbols = returns.columns.tolist()

    return {
        "items": symbols,
        "items_val": last_values[symbols].values,
        "items_returns": returns,
        "items_pred": np.zeros(len(symbols)),  
        "items_vol": np.zeros(len(symbols)),
        "behaviour": behaviour,
        "optimizer": 'markowitz',
        "min_return": min_return
    }

def process_gnosse_data(df, predictions,optimizer='gnosse', behaviour='conservative'):
    df = df.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    predictions = predictions.copy()

    vols = []
    last_values = []

    symbols = sorted(df['symbol'].unique())
    for symbol in symbols:
        df_symbol = df[df['symbol'] == symbol].copy()

        risk = RiskMeasurements(df_symbol)
        vol = risk.historical_volatility()['vol_per_day']
        vols.append(vol)

        last_close = df_symbol.sort_values('date')['close'].dropna().iloc[-1]
        last_values.append(last_close)

    predictions['prediction'] = predictions['prediction'].apply(lambda x: np.array(x)[-1])
    predictions = predictions.sort_values(by='symbol')
    preds = predictions['prediction'].values

    return {
        "items": symbols,
        "items_val": np.array(last_values),
        "items_vol": np.array(vols),
        "items_pred": preds,
        "behaviour": behaviour,
        "items_returns": np.zeros(len(symbols)),
        "min_return": 0,
        "optimizer": optimizer
    }

