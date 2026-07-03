# ai-trading

## Smoke Test

プロジェクトルート `C:/ai-trading` から以下が通ることを確認します。

```bash
python -c "from src.config_loader import load_strategy_config; cfg = load_strategy_config(); print(cfg)"
```
