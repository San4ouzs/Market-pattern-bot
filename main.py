import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import yfinance as yf
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import FSInputFile
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# -------------------------------------------------
#  Загрузка настроек
# -------------------------------------------------

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not BOT_TOKEN:
    raise RuntimeError("Не найден TELEGRAM_BOT_TOKEN в .env файле")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

bot = Bot(BOT_TOKEN)
dp = Dispatcher()


# -------------------------------------------------
#  Вселенная тикеров (пример, можно расширять)
#  Все тикеры должны быть в формате Yahoo Finance
# -------------------------------------------------

UNIVERSE: Dict[str, str] = {
    # Криптовалюты
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "BNB-USD": "BNB",
    "SOL-USD": "Solana",
    "ADA-USD": "Cardano",

    # Фиатные валюты (Forex)
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
    "USDRUB=X": "USD/RUB",
    "USDCHF=X": "USD/CHF",

    # Фондовые индексы
    "^GSPC": "S&P 500",
    "^NDX": "Nasdaq 100",
    "^DJI": "Dow Jones",
    "^RUT": "Russell 2000",
    "^STOXX50E": "Euro Stoxx 50",
    "^GDAXI": "DAX 40",

    # Сырьевые активы (commodities)
    "GC=F": "Gold",
    "SI=F": "Silver",
    "CL=F": "Crude Oil WTI",
    "BZ=F": "Brent Oil",
    "NG=F": "Natural Gas",

    # Акции крупных компаний (пример)
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "META": "Meta Platforms",
}


# -------------------------------------------------
#  Функции загрузки данных
# -------------------------------------------------

def _yf_interval_from_string(interval_str: str) -> str:
    """
    Прямое отображение таймфрейма пользователя в формат Yahoo Finance.
    Допустимые интервалы для пользователя:
    1m, 5m, 15m, 30m, 60m, 1h, 4h, 1d, 1wk, 1mo
    """
    mapping = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "60m": "60m",
        "1h": "60m",
        "4h": "240m",
        "1d": "1d",
        "1D": "1d",
        "1wk": "1wk",
        "1w": "1wk",
        "1W": "1wk",
        "1mo": "1mo",
        "1M": "1mo",
    }
    return mapping.get(interval_str, "1d")


def _calc_start_end(lookback_days: int) -> Tuple[datetime, datetime]:
    """Вычисляем период истории."""
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days + 5)  # небольшой запас
    return start, end


def download_history(
    ticker: str,
    interval: str,
    lookback_days: int,
) -> pd.DataFrame:
    """
    Загрузка исторических данных OHLCV для одного тикера.
    Возвращает DataFrame с колонками: [Open, High, Low, Close, Volume]
    Индекс — datetime (UTC).
    """
    yf_interval = _yf_interval_from_string(interval)
    start, end = _calc_start_end(lookback_days)

    logging.info(f"Загрузка данных {ticker} interval={yf_interval}, {start}..{end}")
    data = yf.download(
        ticker,
        interval=yf_interval,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        logging.warning(f"Пустые данные для {ticker}")
        return pd.DataFrame()

    data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
    data.index = pd.to_datetime(data.index, utc=True)
    return data


def build_return_series(df: pd.DataFrame) -> pd.Series:
    """
    Строим ряд доходностей по Close.
    Используем логарифмические доходности, чтобы лучше сравнивать.
    """
    close = df["Close"]
    returns = np.log(close / close.shift(1))
    return returns.dropna()


def build_volume_change_series(df: pd.DataFrame) -> pd.Series:
    """Ряд изменений объема торговли."""
    vol = df["Volume"]
    chg = vol.pct_change()
    return chg.replace([np.inf, -np.inf], np.nan).dropna()


# -------------------------------------------------
#  Поиск похожих/обратных графиков
# -------------------------------------------------

def calc_correlations(
    base_df: pd.DataFrame, other_df: pd.DataFrame
) -> Tuple[float, float]:
    """
    Считаем корреляцию доходностей и корреляцию изменений объемов.
    Возвращает (corr_price, corr_volume).
    """
    base_ret = build_return_series(base_df)
    other_ret = build_return_series(other_df)

    # Приводим к общему времени
    joined = pd.concat(
        [base_ret.rename("base"), other_ret.rename("other")],
        axis=1,
        join="inner",
    ).dropna()

    if len(joined) < 10:
        return np.nan, np.nan

    corr_price = joined["base"].corr(joined["other"])

    base_vol = build_volume_change_series(base_df)
    other_vol = build_volume_change_series(other_df)

    joined_vol = pd.concat(
        [base_vol.rename("base"), other_vol.rename("other")],
        axis=1,
        join="inner",
    ).dropna()

    if len(joined_vol) < 10:
        corr_vol = np.nan
    else:
        corr_vol = joined_vol["base"].corr(joined_vol["other"])

    return corr_price, corr_vol


def find_similar_assets(
    base_symbol: str,
    interval: str,
    lookback_days: int,
    top_n: int = 5,
    mode: str = "direct",  # "direct" или "inverse"
) -> List[Dict]:
    """
    Находит активы с похожим ("direct") или обратным ("inverse") движением.
    Возвращает список словарей:
    {
        "symbol": str,
        "name": str,
        "corr_price": float,
        "corr_vol": float,
        "base_df": DataFrame,
        "other_df": DataFrame,
    }
    """
    # Загружаем базовый актив
    base_df = download_history(base_symbol, interval, lookback_days)
    if base_df.empty:
        raise ValueError(f"Не удалось загрузить данные для базового актива {base_symbol}")

    results = []
    for symbol, name in UNIVERSE.items():
        if symbol == base_symbol:
            continue

        other_df = download_history(symbol, interval, lookback_days)
        if other_df.empty:
            continue

        corr_price, corr_vol = calc_correlations(base_df, other_df)
        if np.isnan(corr_price):
            continue

        results.append(
            {
                "symbol": symbol,
                "name": name,
                "corr_price": corr_price,
                "corr_vol": corr_vol,
                "base_df": base_df,
                "other_df": other_df,
            }
        )

    if not results:
        return []

    if mode == "inverse":
        # Самые отрицательные корреляции
        results.sort(key=lambda x: x["corr_price"])
    else:
        # Самые положительные корреляции
        results.sort(key=lambda x: x["corr_price"], reverse=True)

    return results[:top_n]


# -------------------------------------------------
#  Построение и сохранение графиков
# -------------------------------------------------

def make_comparison_plot(
    base_symbol: str,
    base_name: str,
    other_symbol: str,
    other_name: str,
    base_df: pd.DataFrame,
    other_df: pd.DataFrame,
    corr_price: float,
    corr_vol: float,
    interval: str,
    lookback_days: int,
    out_path: str,
) -> None:
    """
    Строим график двух активов, нормируем цену к 100 в начале периода.
    """
    # Приводим к общему интервалу
    joined = pd.concat(
        [
            base_df["Close"].rename("base"),
            other_df["Close"].rename("other"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if joined.empty:
        raise ValueError("Нет пересечения данных для построения графика")

    base_norm = joined["base"] / joined["base"].iloc[0] * 100
    other_norm = joined["other"] / joined["other"].iloc[0] * 100

    plt.figure(figsize=(10, 6))
    plt.plot(base_norm.index, base_norm.values, label=f"{base_name} ({base_symbol})")
    plt.plot(other_norm.index, other_norm.values, label=f"{other_name} ({other_symbol})", linestyle="--")

    plt.title(
        f"Сравнение движения: {base_symbol} vs {other_symbol}\n"
        f"corr_price={corr_price:.2f}, corr_volume={corr_vol:.2f} | interval={interval}, lookback={lookback_days}d"
    )
    plt.xlabel("Дата/время (UTC)")
    plt.ylabel("Нормированная цена (100 в начале)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -------------------------------------------------
#  Вспомогательные функции парсинга команд
# -------------------------------------------------

def parse_find_command(text: str) -> Tuple[str, str, int]:
    """
    Парсинг команд /find_like и /find_inverse.
    Ожидается формат:
    /find_like BTC-USD 1d 90
    или
    /find_like BTC-USD 4h 30
    """
    parts = text.strip().split()
    if len(parts) < 4:
        raise ValueError("Нужно передать 3 параметра: <тикер> <таймфрейм> <кол-во_дней>")

    _, base_symbol, interval, days_str = parts[:4]

    try:
        lookback_days = int(days_str)
    except Exception:
        raise ValueError("Количество дней должно быть целым числом")

    return base_symbol, interval, lookback_days


def resolve_name(symbol: str) -> str:
    """Возвращает понятное имя актива, если оно есть в UNIVERSE."""
    return UNIVERSE.get(symbol, symbol)


# -------------------------------------------------
#  Обработчики Telegram-бота
# -------------------------------------------------

HELP_TEXT = (
    "Я бот для поиска похожих и обратных графиков разных рынков.\n\n"
    "Команды:\n"
    "/start — краткая информация\n"
    "/help — список команд\n\n"
    "/find_like <ТИКЕР> <ТАЙМФРЕЙМ> <ДНЕЙ> — найти активы с похожим движением\n"
    "Пример: /find_like BTC-USD 1d 90\n\n"
    "/find_inverse <ТИКЕР> <ТАЙМФРЕЙМ> <ДНЕЙ> — найти активы с обратным движением\n"
    "Пример: /find_inverse BTC-USD 1d 90\n\n"
    "Таймфреймы (формат Yahoo Finance): 1m,5m,15m,30m,60m,1h,4h,1d,1wk,1mo\n"
    "Тикеры в формате Yahoo Finance (BTC-USD, ETH-USD, EURUSD=X, ^GSPC, GC=F, AAPL и т.д.)\n"
)


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    text = (
        "Привет! 👋\n\n"
        "Я анализирую графики цен, объёмов и других индексов для биткоина, валют, "
        "биржевых индексов, сырьевых товаров и акций крупнейших компаний.\n\n"
        "Я могу найти активы с похожим движением или с обратной динамикой "
        "и показать сравнительные графики прямо здесь, в чате.\n\n"
        "Набери /help, чтобы посмотреть список команд."
    )
    await message.answer(text)


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.answer(HELP_TEXT)


async def run_search_and_send(
    message: types.Message,
    mode: str,
):
    """
    Общая логика для /find_like и /find_inverse.
    mode: 'direct' или 'inverse'
    """
    try:
        base_symbol, interval, lookback_days = parse_find_command(message.text)
    except ValueError as e:
        await message.reply(f"Ошибка: {e}\n\nОжидаемый формат:\n/find_like BTC-USD 1d 90")
        return

    base_name = resolve_name(base_symbol)
    mode_text = "похожие" if mode == "direct" else "обратные"

    await message.answer(
        f"Ищу {mode_text} графики для {base_name} ({base_symbol})\n"
        f"Таймфрейм: {interval}, период: {lookback_days} дней.\n"
        f"Загружаю данные с рынка, подождите..."
    )

    loop = asyncio.get_running_loop()

    try:
        results = await loop.run_in_executor(
            None,
            find_similar_assets,
            base_symbol,
            interval,
            lookback_days,
            5,
            mode,
        )
    except Exception as e:
        logging.exception("Ошибка при поиске похожих активов")
        await message.answer(f"Произошла ошибка при анализе: {e}")
        return

    if not results:
        await message.answer("Не удалось найти подходящие активы для заданных параметров.")
        return

    # Создаем временную директорию для картинок
    tmp_dir = "tmp_plots"
    os.makedirs(tmp_dir, exist_ok=True)

    header_lines = [
        f"Результаты для {base_name} ({base_symbol})",
        f"Режим: {'похожие движения' if mode == 'direct' else 'обратные движения'}",
        f"Таймфрейм: {interval}, период: {lookback_days} дней.",
        "",
        "Список найденных активов:",
    ]
    for idx, r in enumerate(results, start=1):
        header_lines.append(
            f"{idx}) {r['name']} ({r['symbol']}): "
            f"corr_price={r['corr_price']:.2f}, corr_volume={r['corr_vol']:.2f}"
        )

    await message.answer("\n".join(header_lines))

    # Генерируем и отправляем графики
    for idx, r in enumerate(results, start=1):
        file_path = os.path.join(tmp_dir, f"compare_{idx}.png")
        try:
            await loop.run_in_executor(
                None,
                make_comparison_plot,
                base_symbol,
                base_name,
                r["symbol"],
                r["name"],
                r["base_df"],
                r["other_df"],
                r["corr_price"],
                r["corr_vol"],
                interval,
                lookback_days,
                file_path,
            )
        except Exception as e:
            logging.exception("Ошибка при построении графика")
            await message.answer(
                f"Не удалось построить график для {r['name']} ({r['symbol']}): {e}"
            )
            continue

        try:
            photo = FSInputFile(file_path)
            caption = (
                f"{idx}) {base_name} ({base_symbol}) vs {r['name']} ({r['symbol']})\n"
                f"corr_price={r['corr_price']:.2f}, corr_volume={r['corr_vol']:.2f}"
            )
            await message.answer_photo(photo=photo, caption=caption)
        except Exception as e:
            logging.exception("Ошибка при отправке графика")
            await message.answer(
                f"Не удалось отправить график для {r['name']} ({r['symbol']}): {e}"
            )


@dp.message(Command("find_like"))
async def cmd_find_like(message: types.Message):
    await run_search_and_send(message, mode="direct")


@dp.message(Command("find_inverse"))
async def cmd_find_inverse(message: types.Message):
    await run_search_and_send(message, mode="inverse")


@dp.message(F.text)
async def fallback_message(message: types.Message):
    """
    Обработчик на все остальные текстовые сообщения.
    """
    text = message.text.strip().lower()
    if text in {"hi", "hello", "привет"}:
        await cmd_start(message)
    else:
        await message.answer(
            "Я не понял команду.\n\n"
            "Используй /help, чтобы увидеть список команд.\n\n"
            "Примеры:\n"
            "/find_like BTC-USD 1d 90\n"
            "/find_inverse BTC-USD 1d 90"
        )


async def main():
    logging.info("Запуск бота...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Бот остановлен.")
