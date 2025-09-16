# do_prediction.py
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 或者添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

from binance.client import Client

from model import Kronos, KronosTokenizer, KronosPredictor
import numpy as np
import torch

def plot_prediction(kline_df, pred_df):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)

    ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()





def fetch_binance_data(symbol, interval):
    """Fetches K-line data from the Binance public API."""
    limit = 300

    print(f"Fetching {limit} bars of {symbol} {interval} data from Binance...")
    client = Client()
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore']
    df = pd.DataFrame(klines, columns=cols)

    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']]
    df.rename(columns={'quote_asset_volume': 'amount', 'open_time': 'timestamps'}, inplace=True)

    df['timestamps'] = pd.to_datetime(df['timestamps'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[col] = pd.to_numeric(df[col])

    print("Data fetched successfully.")
    return df



def main(symbol, iinterval , lookback , pred_len ):
    """
    为什么是42？
    42这个数字在计算机科学和数学领域中具有特殊的文化意义：
    《银河系漫游指南》：在道格拉斯·亚当斯的科幻小说《银河系漫游指南》中，42是"生命、宇宙以及一切"的终极答案
    程序员文化：这个数字在程序员社区中被广泛使用，成为了一种传统和梗
    任意性：实际上任何固定的整数都可以作为随机种子
    :return:
    """
    # 设置随机种子以确保结果可复现
    """     """
    np.random.seed(42)
    torch.manual_seed(42)

    # 如果使用GPU，也设置相应的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1. Load Model and Tokenizer
    tokenizer = KronosTokenizer.from_pretrained("/Users/longquan/Documents/git_repository/myself/kronos/data/outputs/models/finetune_tokenizer_demo/checkpoints/best_model")
    model = Kronos.from_pretrained("/Users/longquan/Documents/git_repository/myself/kronos/data/outputs/models/finetune_predictor_demo/checkpoints/best_model")

    # 2. Instantiate Predictor
    predictor = KronosPredictor(model, tokenizer, device="mps:0", max_context=512)

    # 3. Prepare Data
    df = fetch_binance_data(symbol=symbol, interval=iinterval)

    # 添加数据验证
    if df.empty:
        print(f"错误: 没有找到 symbol={symbol}, iinterval={iinterval} 的数据")
        return

    if len(df) < lookback + pred_len:
        print(f"错误: 数据不足。需要 {lookback + pred_len} 行数据，但只找到 {len(df)} 行")
        return

    print(f"成功加载 {len(df)} 行数据")

    #df['trade_date'] = pd.to_datetime(df['trade_date'])
    #df['volume'] = df['vol']

    x_df = df.loc[:lookback - 1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.loc[:lookback - 1, 'timestamps']
    y_timestamp = df.loc[lookback:lookback + pred_len - 1, 'timestamps']

    # 4. Make Prediction
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True
    )

    # 5. Visualize Results
    # Combine historical and forecasted data for plotting
    kline_df = df.loc[:lookback + pred_len - 1]

    # visualize
    plot_prediction(kline_df, pred_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Kronos Model Prediction')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol (e.g., BTCUSDT)')
    parser.add_argument('--iinterval', type=str, default='15m', help='Interval (e.g., 5m, 15m)')
    parser.add_argument('--lookback', type=int, default=50, help='Lookback period')
    parser.add_argument('--pred_len', type=int, default=20, help='Prediction length')

    args = parser.parse_args()

    print(f"查询参数为：symbol={args.symbol}, iinterval={args.iinterval}, lookback={args.lookback}, pred_len={args.pred_len}")

    main(symbol=args.symbol, iinterval=args.iinterval, lookback=args.lookback, pred_len=args.pred_len)
