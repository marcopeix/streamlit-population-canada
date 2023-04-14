import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import floor
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.forecasting.theta import ThetaModel

st.set_page_config('Forecast', page_icon="🔮")

def smape(actual, predicted):

    if not all([isinstance(actual, np.ndarray), isinstance(predicted, np.ndarray)]):
        actual, predicted = np.array(actual), np.array(predicted)
  
    return round(np.mean(np.abs(predicted - actual) / ((np.abs(predicted) + np.abs(actual))/2))*100, 2)

@st.cache_data
def rolling_predictions(df, train_len, horizon, window, method):
    
    TOTAL_LEN = len(df)
    
    if method == 'ar':
        best_lags = ar_select_order(df[:train_len], maxlag=8, glob=True).ar_lags
        pred_AR = []
        
        for i in range(train_len, TOTAL_LEN, window):
            ar_model = AutoReg(df[:i], lags=best_lags)
            res = ar_model.fit()
            predictions = res.predict(i, i + window -1)
            oos_pred = predictions[-window:]
            pred_AR.extend(oos_pred)
        
        return pred_AR[:horizon]

    elif method == 'holt':
        pred_holt = []
        
        for i in range(train_len, TOTAL_LEN, window):
            des = Holt(df[:i], initialization_method='estimated').fit()
            predictions = des.forecast(window)
            pred_holt.extend(predictions)
            
        return pred_holt[:horizon]
    
    elif method == 'theta':
        pred_theta = []
        
        for i in range(train_len, TOTAL_LEN, window):
            tm = ThetaModel(endog=df[:i], deseasonalize=False)
            res = tm.fit()
            preds = res.forecast(window)
            pred_theta.extend(preds)
            
        return pred_theta[:horizon]

@st.cache_data
def test_and_predict(df, col_name, horizon):
    df = df.reset_index()
    model_list = ['ar', 'holt', 'theta']
    train = df[col_name][:-32]
    test = df[['Quarter', col_name]][-32:]
    total_len = len(df)
    
    train_len = len(train)
    test_len = len(test)
    
    pred_AR = rolling_predictions(df[col_name], train_len, test_len, horizon, 'ar')
    pred_holt = rolling_predictions(df[col_name], train_len, test_len, horizon, 'holt')
    pred_theta = rolling_predictions(df[col_name], train_len, test_len, horizon, 'theta')
    
    test['pred_AR'] = pred_AR
    test['pred_holt'] = pred_holt
    test['pred_theta'] = pred_theta
    
    smapes = []
    
    smapes.append(smape(test[col_name], test['pred_AR']))
    smapes.append(smape(test[col_name], test['pred_holt']))    
    smapes.append(smape(test[col_name], test['pred_theta']))    
    
    best_model = model_list[np.argmin(smapes)]
    
    if best_model == 'ar':
        best_lags = ar_select_order(train, maxlag=8, glob=True).ar_lags
        ar_model = AutoReg(df[col_name], lags=best_lags)
        res = ar_model.fit()
        predictions = res.predict(total_len, total_len + horizon - 1)
        
        return predictions, smapes
    
    elif best_model == 'holt':
        des = Holt(df[col_name], initialization_method='estimated').fit()
        predictions = des.forecast(horizon)
        
        return predictions, smapes
    
    elif best_model == 'theta':
        tm = ThetaModel(endog=df[col_name], deseasonalize=False)
        res = tm.fit()
        predictions = res.forecast(horizon)
        
        return predictions, smapes        



st.title('Forecast the Quarterly Population in Canada')

df = st.session_state['df']

col1, col2 = st.columns(2)
target = col1.selectbox('Select your target', df.columns)
horizon = col2.slider('Choose the horizon', min_value=1, max_value=16, value=1, step=1)
forecast_btn = st.button('Forecast')

if forecast_btn:
    preds, smapes = test_and_predict(df, target, horizon)

    tab1, tab2 = st.tabs(['Predictions', 'Model evaluation'])
    pred_fig, pred_ax = plt.subplots()
    pred_ax.plot(df[target])
    pred_ax.plot(preds, label='Forecast')
    pred_ax.set_xlabel('Time')
    pred_ax.set_ylabel('Population')
    pred_ax.legend(loc=2)
    pred_ax.set_xticks(np.arange(2, len(df) + len(preds), 8))
    pred_ax.set_xticklabels(np.arange(1992, 2024 + floor(len(preds)/4), 2))
    pred_fig.autofmt_xdate()
    tab1.pyplot(pred_fig)

    eval_fig, eval_ax = plt.subplots()

    x = ['AR', 'DES', 'Theta']
    y = smapes

    eval_ax.bar(x, y, width=0.4)
    eval_ax.set_xlabel('Models')
    eval_ax.set_ylabel('sMAPE')
    eval_ax.set_ylim(0, max(smapes)+0.1)

    for index, value in enumerate(y):
        plt.text(x=index, y=value + 0.015, s=str(round(value,2)), ha='center')

    tab2.pyplot(eval_fig)

expander = st.expander("How does it work 🤓")
expander.write("""
After specifying the target and horizon, three models are tested: an autoregressive model, double exponential smoothing, and the Theta model.

Each model is tested on a hold-out set of 32 timesteps to evaluate their performance in predicting on the set horizon for the specified target. The evaluation is done using the sMAPE.

Then, the best model is the one that achieves the lowest sMAPE. 

That model is automatically selected to generate the predictions.
""")
