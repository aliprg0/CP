import tensorflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tvDatafeed import TvDatafeed, Interval
import yfinance as yf
from tensorflow.keras.models import load_model
import os
from datetime import datetime


def load_models(timeframe):
    models = {}
    # load all models in the models folder
    for root, dirs, files in os.walk("./models"):
        for file in files:
            if file.endswith(".h5"):
                models[file] = load_model(os.path.join(root, file))
    return models


def scaler(row):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    row = scaler.fit_transform(row)
    return row


def process_for_prediction(data, index):
    i = index
    if "symbol" in data.columns:
        data.drop("symbol", axis=1, inplace=True)
    if "datetime" in data.columns:
        data.drop("datetime", axis=1, inplace=True)
    if "Adj Close" in data.columns:
        data.drop("Adj Close", axis=1, inplace=True)
    data = np.array(data)
    grow = []
    ggggrow = []
    for x in range(1, 31):
        grow.append([data[i-x][0] - data[i-(1+x)][0]])
        ggggrow.append([data[i-x][3] - data[i-(1+x)][3]])
    arr = np.array(grow).flatten()
    arr4 = np.array(ggggrow).flatten()
    arr = np.concatenate((arr, arr4), axis=0).reshape(-1, 1)
    arr = scaler(arr.reshape(-1, 1))
    return arr


def make_prediction_for_yf(symbol, period, timeframe, index,models,special_models,suggested_models):
    models_reverse = {v: k for k, v in models.items()}
    data = yf.download(symbol, period=period,
                       interval=timeframe, progress=False)
    raw_data = process_for_prediction(data, index)
    for model in models_reverse:
        prediction = model.predict(np.array(raw_data).reshape(1, -1))
        yresults.append(
            f"{models_reverse.get(model)} : {prediction}")
        y_only_num.append(prediction)
    special_models_reverse = {v: k for k, v in special_models.items()}
    for model in special_models_reverse:
        prediction = model.predict(np.array(raw_data).reshape(1, -1))
        special_models_yresults.append(
            f"{special_models_reverse.get(model)} : {prediction}")
        special_models_y_only_num.append(prediction)
    suggested_models_reverse = {v: k for k, v in suggested_models.items()}
    for model in suggested_models_reverse:
        prediction = model.predict(np.array(raw_data).reshape(1, -1))
        suggested_models_yresults.append(
            f"{suggested_models_reverse.get(model)} : {prediction}")
        suggested_models_y_only_num.append(prediction)


def make_prediction_for_tv(symbol, exchange, timeframe, tindex, models, special_models, suggested_models):
    models_reverse = {v: k for k, v in models.items()}
    tv = TvDatafeed()
    raw_data = process_for_prediction(tv.get_hist(
        symbol=symbol, exchange=exchange, interval=timeframe, n_bars=33), tindex)
    for model in models_reverse:
        prediction = model.predict(np.array(raw_data).reshape(1, -1))
        tresults.append(
            f"{models_reverse.get(model)} : {prediction}")
        t_only_num.append(prediction)
    special_models_reverse = {v: k for k, v in special_models.items()}
    for model in special_models_reverse:
        prediction = model.predict(np.array(raw_data).reshape(1, -1))
        special_models_tresults.append(
            f"{special_models_reverse.get(model)} : {prediction}")
        special_models_t_only_num.append(prediction)
    suggested_models_reverse = {v: k for k, v in suggested_models.items()}
    for model in suggested_models_reverse:
        prediction = model.predict(np.array(raw_data).reshape(1, -1))
        suggested_models_tresults.append(
            f"{suggested_models_reverse.get(model)} : {prediction}")
        suggested_models_t_only_num.append(prediction)

def make_prediction(ysymbol, period, timeframe, tsymbol, texchange, ttimeframe, tindex, index, tsymbol2,models,special_models,suggested_models):
    make_prediction_for_yf(ysymbol, period, timeframe, index,models,special_models=special_models,suggested_models=suggested_models)
    try:
        make_prediction_for_tv(tsymbol, texchange, ttimeframe, tindex,models,special_models,suggested_models)
    except:
        make_prediction_for_tv(tsymbol2, texchange, ttimeframe, tindex,models,special_models,suggested_models)


def interval_handler(timeframe):
    if timeframe == "15m":
        return Interval.in_15_minute
    elif timeframe == "1h":
        return Interval.in_1_hour
    elif timeframe == "1d":
        return Interval.in_daily
    elif timeframe == "1wk":
        return Interval.in_weekly
    elif timeframe == "1mo":
        return Interval.in_monthly


def period_handler(period):
    if period == "15m":
        return "10d"
    elif period == "1h":
        return "10d"
    elif period == "1d":
        return "50d"
    elif period == "1wk":
        return "max"
    elif period == "1mo":
        return "max"


def symbol_handler(symbol):
    return f"{symbol}-usd", f"{symbol}usd", f"{symbol}usdt"


def yf_data_handler(symbol, period, timeframe):
    data = yf.download(symbol, period=period,
                       interval=timeframe, progress=False)
    index_c = data.index.values
    index = index_c[-3:]
    if timeframe == "15m":
        mins = []
        for i in index:
            mins.append(str(i).split(":")[1])
        for i in mins:
            if int(i) % 15 != 0:
                return -2, index
        return -1, index
    if timeframe == "1h":
        if str(index[-1]).strip().split(":")[0][-2:] == str(index[-2]).strip().split(":")[0][-2:]:
            return -2, index
        else:
            return -1, index
    if timeframe == "1d":
        return -1, index
    if timeframe == "1wk":
        if int(str(index[-2]).strip().split("-")[-1][:2]) - int(str(index[-1]).strip().split("-")[-1][:2]) != 7:
            return -2, index
        else:
            return -1, index
    if timeframe == "1mo":
        if int(str(index[-2]).split("-")[-2]) == int(str(index[-1]).split("-")[-2]):
            return -2, index
        else:
            return -1, index
def get_special_models_names(timeframe):
    if timeframe == "15m":
        return "15SG136.h5", "15CG136.h5"
    elif timeframe == "1h":
        return "HSG136.h5", "HCG136.h5"
    elif timeframe == "1d":
        return "DSG136.h5", "DCG136.h5"
    elif timeframe == "1wk":
        return "WSG136.h5", "WCG136.h5"
    elif timeframe == "1mo":
        return "MSG136.h5", "MCG136.h5"

def load_suggested_models():
    models = {}
    #load models in 1mo - 1wk - 1d
    lst = os.listdir(os.getcwd()+"/models")
    models = {}
    # load models in 1mo - 1wk - 1d
    # drop 1h and 15m from the list
    lst = [i for i in lst if i not in ["15SG136.h5", "15CG136.h5", "HSG136.h5", "HCG136.h5"]]
    for i in lst:
        models_in_folder = os.listdir(os.getcwd()+"/models/"+i)
        for j in models_in_folder:
            models[j] = load_model(os.getcwd()+"/models/"+i+"/"+j)
    return models
    
def predict(symbol, timeframe):
    global yresults
    global tresults
    global models_reverse
    global y_only_num
    global t_only_num
    global normal_models_prediction
    global special_models_prediction
    global special_models_yresults
    global special_models_tresults
    global special_models_y_only_num
    global special_models_t_only_num
    global suggested_models_yresults
    global suggested_models_tresults
    global suggested_models_y_only_num
    global suggested_models_t_only_num

    models = load_models(timeframe)
    suggested_models = load_suggested_models()
    special_models = get_special_models_names(timeframe)
    #copy models with special names into another dict
    special_models_dict = {}
    for i in special_models:
        special_models_dict[i] = models.pop(i)

    yresults = []
    tresults = []
    y_only_num = []
    t_only_num = []
    special_models_yresults = []
    special_models_tresults = []
    special_models_y_only_num = []
    special_models_t_only_num = []
    suggested_models_yresults = []
    suggested_models_tresults = []
    suggested_models_y_only_num = []
    suggested_models_t_only_num = []
    infos = []
    normal_models_prediction = []
    special_models_prediction = []
    exchange = "Binance"

    ttimeframe = interval_handler(timeframe)
    period = period_handler(timeframe)
    ysymbol, tsymbol, tsymbol2 = symbol_handler(symbol)

    tindex = -1
    index, last_three = yf_data_handler(ysymbol, period, timeframe)

    make_prediction(ysymbol, period, timeframe, tsymbol,
                    exchange, ttimeframe, tindex, index, tsymbol2,models,special_models_dict,suggested_models)

    # all models
    all_y = y_only_num + special_models_y_only_num
    y_buy = 0
    y_sell = 0
    for i in all_y:
        if list(i)[0][0] > list(i)[0][1]:
            y_buy += 1
        else:
            y_sell += 1

    
    all_t = t_only_num + special_models_t_only_num
    t_buy = 0
    t_sell = 0
    for i in all_t:
        if list(i)[0][0] > list(i)[0][1]:
            t_buy += 1
        else:
            t_sell += 1


    all_buy = y_buy + t_buy
    all_sell = y_sell + t_sell
    plus = all_buy - all_sell


    # special models
    special_y = special_models_y_only_num
    special_t = special_models_t_only_num
    special_y_buy = 0
    special_y_sell = 0
    special_t_buy = 0
    special_t_sell = 0
    special_y_buy_percent = 0
    special_y_sell_percent = 0
    special_t_buy_percent = 0
    special_t_sell_percent = 0
    special_all_buy = 0
    special_all_sell = 0
    special_plus = 0

    for i in special_y:
        if list(i)[0][0] > list(i)[0][1]:
            special_y_buy += 1
        else:
            special_y_sell += 1
    for i in special_t:
        if list(i)[0][0] > list(i)[0][1]:
            special_t_buy += 1
        else:
            special_t_sell += 1
    special_all_buy = special_y_buy + special_t_buy
    special_all_sell = special_y_sell + special_t_sell
    special_plus = special_all_buy - special_all_sell

    # suggested models
    suggested_y = suggested_models_y_only_num
    suggested_t = suggested_models_t_only_num
    suggested_y_buy = 0
    suggested_y_sell = 0
    suggested_t_buy = 0
    suggested_t_sell = 0
    suggested_y_buy_percent = 0
    suggested_y_sell_percent = 0
    suggested_t_buy_percent = 0
    suggested_t_sell_percent = 0
    suggested_all_buy = 0
    suggested_all_sell = 0
    suggested_plus = 0

    for i in suggested_y:
        if list(i)[0][0] > list(i)[0][1]:
            suggested_y_buy += 1
        else:
            suggested_y_sell += 1
    for i in suggested_t:
        if list(i)[0][0] > list(i)[0][1]:
            suggested_t_buy += 1
        else:
            suggested_t_sell += 1

    suggested_all_buy = suggested_y_buy + suggested_t_buy
    suggested_all_sell = suggested_y_sell + suggested_t_sell
    suggested_plus = suggested_all_buy - suggested_all_sell
   

    suggestion = plus + special_plus + suggested_plus 
    if suggestion > 0:
        suggestion = f"BUY {suggestion}/{all_buy+special_all_buy+suggested_all_buy+all_sell+special_all_sell+suggested_all_sell}"
    elif suggestion < 0:
        suggestion = f"SELL {-1 * suggestion}/{all_buy+special_all_buy+suggested_all_buy+all_sell+special_all_sell+suggested_all_sell}"
    else:
        suggestion = f"NEUTRAL"
    
    # only yahoo finance suggestion
    y_suggestion = y_buy + special_y_buy + suggested_y_buy - y_sell - special_y_sell - suggested_y_sell
    if y_suggestion > 0:
        y_suggestion = f"BUY {y_suggestion}/{y_buy+special_y_buy+suggested_y_buy+y_sell+special_y_sell+suggested_y_sell}"
    elif y_suggestion < 0:
        y_suggestion = f"SELL {-1 * y_suggestion}/{y_buy+special_y_buy+suggested_y_buy+y_sell+special_y_sell+suggested_y_sell}"
    else:
        y_suggestion = f"NEUTRAL"
    # only tv suggestion
    t_suggestion = t_buy + special_t_buy + suggested_t_buy - t_sell - special_t_sell - suggested_t_sell
    if t_suggestion > 0:
        t_suggestion = f"BUY {t_suggestion}/{t_buy+special_t_buy+suggested_t_buy+t_sell+special_t_sell+suggested_t_sell}"
    elif t_suggestion < 0:
        t_suggestion = f"SELL {-1 * t_suggestion}/{t_buy+special_t_buy+suggested_t_buy+t_sell+special_t_sell+suggested_t_sell}"
    else:
        t_suggestion = f"NEUTRAL"
    

    # info for all information
    infos.append(f'''
Prediction for {symbol.upper()}
Timeframe: {timeframe}
YF index: {index}
TV index: {tindex}
Current_utc_time: {str(datetime.utcnow())}
Last_three: {last_three}
---------------------------
Y_Results: {yresults}
T_Results: {tresults}
----------------------------
All Models:
Yf_buy: {y_buy}
Yf_sell: {y_sell}
Tv_buy: {t_buy}
Tv_sell: {t_sell}
All_buy: {all_buy}
All_sell: {all_sell}
All: {plus}
----------------------------
Special Models({timeframe}) :
Yf_buy: {special_y_buy}
Yf_sell: {special_y_sell}
Tv_buy: {special_t_buy}
Tv_sell: {special_t_sell}
All_buy: {special_all_buy}
All_sell: {special_all_sell}
All: {special_plus}
----------------------------
Suggested Models(Monthly, Weekly, Daily) :
Yf_buy: {suggested_y_buy}
Yf_sell: {suggested_y_sell}
Tv_buy: {suggested_t_buy}
Tv_sell: {suggested_t_sell}
All_buy: {suggested_all_buy}
All_sell: {suggested_all_sell}
All: {suggested_plus}
----------------------------
YF_Suggestion : {y_suggestion}
All_Suggestion : {suggestion}
TV_Suggestion : {t_suggestion}
''')

    yresults = []
    tresults = []
    y_only_num = []
    t_only_num = []
    special_models_yresults = []
    special_models_tresults = []
    special_models_y_only_num = []
    special_models_t_only_num = []
    suggested_models_yresults = []
    suggested_models_tresults = []
    suggested_models_y_only_num = []
    suggested_models_t_only_num = []
    normal_models_prediction = []
    special_models_prediction = []

    ttimeframe = interval_handler(timeframe)
    period = period_handler(timeframe)
    ysymbol, tsymbol, tsymbol2 = symbol_handler(symbol)

    tindex = -1
    index, last_three = yf_data_handler(ysymbol, period, timeframe)

    tindex = 0
    index = index + 1

    make_prediction(ysymbol, period, timeframe, tsymbol,
                    exchange, ttimeframe, tindex, index, tsymbol2, models, special_models_dict, suggested_models)

    # all models
    all_y = y_only_num + special_models_y_only_num
    y_buy = 0
    y_sell = 0
    for i in all_y:
        if list(i)[0][0] > list(i)[0][1]:
            y_buy += 1
        else:
            y_sell += 1


    all_t = t_only_num + special_models_t_only_num
    t_buy = 0
    t_sell = 0
    for i in all_t:
        if list(i)[0][0] > list(i)[0][1]:
            t_buy += 1
        else:
            t_sell += 1


    all_buy = y_buy + t_buy
    all_sell = y_sell + t_sell
    plus = all_buy - all_sell

    # special models
    special_y = special_models_y_only_num
    special_t = special_models_t_only_num
    special_y_buy = 0
    special_y_sell = 0
    special_t_buy = 0
    special_t_sell = 0
    special_y_buy_percent = 0
    special_y_sell_percent = 0
    special_t_buy_percent = 0
    special_t_sell_percent = 0
    special_all_buy = 0
    special_all_sell = 0
    special_plus = 0

    for i in special_y:
        if list(i)[0][0] > list(i)[0][1]:
            special_y_buy += 1
        else:
            special_y_sell += 1
    for i in special_t:
        if list(i)[0][0] > list(i)[0][1]:
            special_t_buy += 1
        else:
            special_t_sell += 1
    special_all_buy = special_y_buy + special_t_buy
    special_all_sell = special_y_sell + special_t_sell
    special_plus = special_all_buy - special_all_sell

    # suggested models
    suggested_y = suggested_models_y_only_num
    suggested_t = suggested_models_t_only_num
    suggested_y_buy = 0
    suggested_y_sell = 0
    suggested_t_buy = 0
    suggested_t_sell = 0
    suggested_y_buy_percent = 0
    suggested_y_sell_percent = 0
    suggested_t_buy_percent = 0
    suggested_t_sell_percent = 0
    suggested_all_buy = 0
    suggested_all_sell = 0
    suggested_plus = 0

    for i in suggested_y:
        if list(i)[0][0] > list(i)[0][1]:
            suggested_y_buy += 1
        else:
            suggested_y_sell += 1
    for i in suggested_t:
        if list(i)[0][0] > list(i)[0][1]:
            suggested_t_buy += 1
        else:
            suggested_t_sell += 1
    suggested_all_buy = suggested_y_buy + suggested_t_buy
    suggested_all_sell = suggested_y_sell + suggested_t_sell
    suggested_plus = suggested_all_buy - suggested_all_sell

    suggestion = plus + special_plus + suggested_plus
    if suggestion > 0:
        suggestion = f"BUY {suggestion}/{all_buy+special_all_buy+suggested_all_buy+all_sell+special_all_sell+suggested_all_sell}"
    elif suggestion < 0:
        suggestion = f"SELL {-1 * suggestion}/{all_buy+special_all_buy+suggested_all_buy+all_sell+special_all_sell+suggested_all_sell}"
    else:
        suggestion = f"NEUTRAL"

    # only yahoo finance suggestion
    y_suggestion = y_buy + special_y_buy + suggested_y_buy - \
        y_sell - special_y_sell - suggested_y_sell
    if y_suggestion > 0:
        y_suggestion = f"BUY {y_suggestion}/{y_buy+special_y_buy+suggested_y_buy+y_sell+special_y_sell+suggested_y_sell}"
    elif y_suggestion < 0:
        y_suggestion = f"SELL {-1 * y_suggestion}/{y_buy+special_y_buy+suggested_y_buy+y_sell+special_y_sell+suggested_y_sell}"
    else:
        y_suggestion = f"NEUTRAL"
    # only tv suggestion
    t_suggestion = t_buy + special_t_buy + suggested_t_buy - \
        t_sell - special_t_sell - suggested_t_sell
    if t_suggestion > 0:
        t_suggestion = f"BUY {t_suggestion}/{t_buy+special_t_buy+suggested_t_buy+t_sell+special_t_sell+suggested_t_sell}"
    elif t_suggestion < 0:
        t_suggestion = f"SELL {-1 * t_suggestion}/{t_buy+special_t_buy+suggested_t_buy+t_sell+special_t_sell+suggested_t_sell}"
    else:
        t_suggestion = f"NEUTRAL"

    # info for all information
    infos.append(f'''
Future Prediction for {symbol.upper()}
Timeframe: {timeframe}
YF index: {index}
TV index: {tindex}
Current_utc_time: {str(datetime.utcnow())}
Last_three: {last_three}
---------------------------
Y_Results: {yresults}
T_Results: {tresults}
----------------------------
All Models:
Yf_buy: {y_buy}
Yf_sell: {y_sell}
Tv_buy: {t_buy}
Tv_sell: {t_sell}
All_buy: {all_buy}
All_sell: {all_sell}
All: {plus}
----------------------------
Special Models({timeframe}) :
Yf_buy: {special_y_buy}
Yf_sell: {special_y_sell}
Tv_buy: {special_t_buy}
Tv_sell: {special_t_sell}
All_buy: {special_all_buy}
All_sell: {special_all_sell}
All: {special_plus}
----------------------------
Suggested Models(Monthly, Weekly, Daily) :
Yf_buy: {suggested_y_buy}
Yf_sell: {suggested_y_sell}
Tv_buy: {suggested_t_buy}
Tv_sell: {suggested_t_sell}
All_buy: {suggested_all_buy}
All_sell: {suggested_all_sell}
All: {suggested_plus}
----------------------------
YF_Suggestion : {y_suggestion}
All_Suggestion : {suggestion}
TV_Suggestion : {t_suggestion}
''')


    return infos



if __name__ == "__main__":

    info = predict("btc", "1d")
    print(info[0])
    print(info[1])
