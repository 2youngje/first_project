from datasets import load_dataset
import pandas as pd
import os

def fetch_exchange_data_from_huggingface():
    print("[*] Hugging Face에서 환율 데이터 불러오는 중...")
    dataset = load_dataset("hkunlp/international-exchange-rate", split="train")
    df = pd.DataFrame(dataset)

    # 날짜 포맷 정리
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", "Currency", "Rate"]]  # 필요한 열만 추출

    # 저장 위치 설정
    output_dir = "../data/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "exchange_rates.csv")
    df.to_csv(output_path, index=False)

    print(f"[✓] 데이터 저장 완료: {output_path}")

if __name__ == "__main__":
    fetch_exchange_data_from_huggingface()