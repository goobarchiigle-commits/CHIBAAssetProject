import matplotlib.pyplot as plt
import platform

# ウィンドウズ環境等での日本語文字化け対策
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'MS Gothic'
elif platform.system() == 'Darwin': # Mac
    plt.rcParams['font.family'] = 'AppleGothic'

def simulate_assets(current_age, current_savings, monthly_contribution, annual_yield, target_age=80):
    ages = []
    assets = []
    
    current_asset = current_savings
    # 月利の計算
    monthly_rate = annual_yield / 100 / 12
    
    for age in range(current_age, target_age + 1):
        ages.append(age)
        assets.append(current_asset)
        
        # 1年分(12ヶ月)の複利計算
        for _ in range(12):
            current_asset += monthly_contribution
            current_asset *= (1 + monthly_rate)
            
    return ages, assets

def main():
    print("=== 将来の資産推移シミュレーション ===")
    
    # ユーザーからの入力受け付け
    try:
        current_age = int(input("現在の年齢を入力してください: "))
        current_savings = float(input("現在の貯金額（万円）を入力してください: "))
        monthly_contribution = float(input("毎月の積立額（万円）を入力してください: "))
        annual_yield = float(input("想定利回り（年利 %）を入力してください: "))
    except ValueError:
        print("\n[!] 入力が数値として正しくありません。以下のデフォルト値で計算します。")
        current_age = 30
        current_savings = 500
        monthly_contribution = 5
        annual_yield = 5.0
        print(f"現在の年齢: {current_age}歳, 現在の貯金額: {current_savings}万円, 毎月の積立額: {monthly_contribution}万円, 想定利回り: {annual_yield}%\n")

    target_ages_to_check = [50, 60]
    max_age_to_plot = max(80, current_age + 30) # 少なくとも30年後か80歳までは描画
    
    # シミュレーション実行
    ages, assets = simulate_assets(current_age, current_savings, monthly_contribution, annual_yield, target_age=max_age_to_plot)
    
    print("\n--- シミュレーション結果 ---")
    for check_age in target_ages_to_check:
        if check_age >= current_age:
            index = check_age - current_age
            amount = assets[index]
            print(f"■ {check_age}歳の時の想定資産額: 約 {amount:,.1f} 万円")
        else:
            print(f"■ {check_age}歳は現在の年齢よりも低いため計算できません。")
            
    # ------ グラフの描画 ------
    plt.figure(figsize=(10, 6))
    plt.plot(ages, assets, marker='', linestyle='-', linewidth=2, color='#1f77b4', label='資産推移')
    
    plt.title("将来の資産推移シミュレーション", fontsize=16)
    plt.xlabel("年齢 (歳)", fontsize=12)
    plt.ylabel("資産額 (万円)", fontsize=12)
    
    # 50歳と60歳のポイントを強調
    for check_age in target_ages_to_check:
        if check_age >= current_age and check_age <= max_age_to_plot:
            index = check_age - current_age
            amount = assets[index]
            # ポイントを赤色で強調表示
            plt.plot(check_age, amount, marker='o', color='red', markersize=8)
            # テキストによる注釈を付与
            plt.annotate(f"{check_age}歳\n約{amount:,.0f}万円", 
                         (check_age, amount), 
                         textcoords="offset points", 
                         xytext=(0, 15), 
                         ha='center',
                         fontsize=11,
                         color='red',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
            
    # 背景のグリッドとレイアウト調整
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.fill_between(ages, assets, alpha=0.2, color='#1f77b4') # グラフの下を薄く塗りつぶして視認性アップ
    plt.legend()
    plt.tight_layout()
    
    print("\nグラフを表示しています。グラフのウィンドウを閉じるとプログラムが終了します。")
    plt.show()

if __name__ == "__main__":
    main()
