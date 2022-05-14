import logging
from main import predict
from telegram import ForceReply, Update,Bot
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
import time

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


symbols = ["btc","eth","trx","ltc","xrp"]
timeframes = ["15m","1h","1d","1wk","1mo"]

   
def start() :
    bot = Bot('5360333591: AAENhpg7oisn1Gho0C40zY4HebV34NK-8YY')
    while True:
      for i in symbols:
        for j in timeframes:
          bot.send_message("@getcryptupdate", predict(i, j,future_prediction=False, using_all_models=True))
          time.sleep(120)


if __name__ == "__main__":
  start()
