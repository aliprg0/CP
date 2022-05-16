import logging
from main import predict,load_models,load_special_models,load_suggested_models
from telegram import Bot
import time

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


symbols = ["btc","eth","trx","ltc","xrp","bnb","ada","sql","doge","dot","shib",""]
timeframes = ["1d","1wk","1mo"]

   
def start() :
    bot = Bot('5360333591:AAENhpg7oisn1Gho0C40zY4HebV34NK-8YY')
    models = load_models()
    suggested_models = load_suggested_models(models)
    while True:
      for i in symbols:
        for j in timeframes:
          try:
             special_models = load_special_models(models,j)
             bot.send_message("@getcryptupdate",predict(i,j,
                              models,special_models,suggested_models))
             time.sleep(10)
             bot.send_message("@getcryptupdate", predict(i, j,
                              models, special_models, suggested_models))
             time.sleep(30)
          except Exception as e:
             bot.send_message("@getcryptupdate", "Error: " + str(e))

if __name__ == "__main__":
  start()
