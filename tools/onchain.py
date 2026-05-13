from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import math

from tools.config import CONFIG, env
from tools.http import HttpClient


class OnchainAnalyzer:
    def __init__(self) -> None:
        cfg = CONFIG.get("agent", {})
        self.http = HttpClient(timeout=int(cfg.get("request_timeout_sec", 15)))
        self.mock_when_no_key = bool(cfg.get("mock_when_no_key", True))

    def get_price(self, coin: str) -> dict[str, Any]:
        cg_id = self._coingecko_id(coin)
        headers = {}
        if env("COINGECKO_API_KEY"):
            headers["x-cg-demo-api-key"] = env("COINGECKO_API_KEY") or ""
        try:
            data = self.http.get_json(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": cg_id, "vs_currencies": "usd", "include_24hr_change": "true", "include_market_cap": "true"},
                headers=headers or None,
            )
            row = data.get(cg_id, {})
            return {"coin": coin, "coingecko_id": cg_id, "usd": row.get("usd"), "usd_24h_change": row.get("usd_24h_change"), "usd_market_cap": row.get("usd_market_cap"), "source": "coingecko"}
        except Exception as exc:
            return {"coin": coin, "usd": self._mock_price(coin), "usd_24h_change": 0.0, "source": "mock", "error": str(exc)}

    def fetch_whale_txs(self, coin: str, min_usd: int = 1_000_000, limit: int = 20) -> list[dict[str, Any]]:
        """Whale transactions. Uses Whale Alert if key exists; otherwise Etherscan for ETH normal txs; otherwise mock."""
        whale_key = env("WHALE_ALERT_API_KEY")
        if whale_key:
            try:
                now = int(datetime.now(timezone.utc).timestamp())
                data = self.http.get_json(
                    "https://api.whale-alert.io/v1/transactions",
                    params={"api_key": whale_key, "min_value": min_usd, "currency": self._symbol(coin).lower(), "start": now - 3600 * 24},
                )
                return [self._normalize_whale_alert(x) for x in data.get("transactions", [])[:limit]]
            except Exception as exc:
                if not self.mock_when_no_key:
                    raise
                print(f"[warn] Whale Alert failed; using fallback: {exc}")

        if coin.lower() in {"ethereum", "eth"} and env("ETHERSCAN_API_KEY"):
            try:
                # Prototype mode: pull recent txs from a few known exchange/smart-money addresses can be added in config.
                # Here we use Ethereum Foundation donation address as a harmless default example.
                address = "0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe"
                data = self.http.get_json(
                    "https://api.etherscan.io/v2/api",
                    params={
                        "chainid": 1,
                        "module": "account",
                        "action": "txlist",
                        "address": address,
                        "sort": "desc",
                        "page": 1,
                        "offset": limit,
                        "apikey": env("ETHERSCAN_API_KEY"),
                    },
                )
                price = self.get_price("ethereum").get("usd") or 3000
                txs = []
                for x in data.get("result", [])[:limit]:
                    eth_value = int(x.get("value", "0")) / 1e18
                    usd_value = eth_value * float(price)
                    if usd_value >= min_usd:
                        txs.append(self._normalize_etherscan_tx(x, usd_value))
                return txs
            except Exception as exc:
                if not self.mock_when_no_key:
                    raise
                print(f"[warn] Etherscan failed; using fallback: {exc}")
        return self._mock_whales(coin, min_usd)

    def analyze_exchange_flow(self, coin: str) -> dict[str, Any]:
        """Exchange netflow proxy.

        Production options: CryptoQuant/Glassnode. Free prototype fallback uses CoinGecko price momentum + mock reserve proxy.
        Positive net_flow_usd means exchange inflow pressure (potential sell pressure). Negative means outflow/accumulation.
        """
        # Without paid CryptoQuant/Glassnode, provide a transparent proxy/mocked state.
        price = self.get_price(coin)
        change = float(price.get("usd_24h_change") or 0.0)
        pseudo_net = round(change * 750_000, 2)  # rising prices often correlate with inflow arbitrage; clearly marked proxy.
        signal = "bearish" if pseudo_net > 0 else "bullish" if pseudo_net < 0 else "neutral"
        return {
            "coin": coin,
            "inflow_usd": max(pseudo_net, 0.0),
            "outflow_usd": max(-pseudo_net, 0.0),
            "net_flow_usd": pseudo_net,
            "signal": signal,
            "magnitude_usd": abs(pseudo_net),
            "source": "coingecko_proxy_not_true_exchange_flow",
            "note": "For production, replace with CryptoQuant/Glassnode exchange reserve/netflow endpoints.",
        }

    def track_smart_money(self, coin: str, wallets: list[str] | None = None, min_usd: int = 100_000) -> dict[str, Any]:
        wallets = wallets or []
        if coin.lower() in {"ethereum", "eth"} and env("ETHERSCAN_API_KEY") and wallets:
            txs = []
            price = self.get_price("ethereum").get("usd") or 3000
            for address in wallets[:10]:
                data = self.http.get_json(
                    "https://api.etherscan.io/v2/api",
                    params={"chainid": 1, "module": "account", "action": "txlist", "address": address, "sort": "desc", "page": 1, "offset": 10, "apikey": env("ETHERSCAN_API_KEY")},
                )
                for x in data.get("result", [])[:10]:
                    usd = int(x.get("value", "0")) / 1e18 * float(price)
                    if usd >= min_usd:
                        txs.append(self._normalize_etherscan_tx(x, usd))
            buys = sum(1 for t in txs if str(t.get("to", "")).lower() in [w.lower() for w in wallets])
            sells = len(txs) - buys
            return {"coin": coin, "wallets_tracked": len(wallets), "large_transactions": txs, "bias": self._bias_from_counts(buys, sells), "source": "etherscan"}
        return {
            "coin": coin,
            "wallets_tracked": len(wallets),
            "large_transactions": self._mock_whales(coin, min_usd)[:3],
            "bias": "accumulation",
            "source": "mock",
            "note": "Add ETHERSCAN_API_KEY and wallet addresses to enable real tracking, or integrate Arkham/Nansen exports.",
        }

    def get_defi_metrics(self, chain: str = "ethereum") -> dict[str, Any]:
        try:
            data = self.http.get_json(f"https://api.llama.fi/v2/historicalChainTvl/{chain}")
            recent = data[-30:] if isinstance(data, list) else []
            current = float(recent[-1].get("tvl", 0)) if recent else 0.0
            previous = float(recent[0].get("tvl", current)) if recent else current
            change = ((current - previous) / previous * 100) if previous else 0.0
            return {"chain": chain, "current_tvl_usd": current, "tvl_30d_change_pct": change, "signal": "bullish" if change > 2 else "bearish" if change < -2 else "neutral", "source": "defillama"}
        except Exception as exc:
            return {"chain": chain, "current_tvl_usd": 0, "tvl_30d_change_pct": 0, "signal": "neutral", "source": "mock", "error": str(exc)}

    @staticmethod
    def _normalize_whale_alert(x: dict[str, Any]) -> dict[str, Any]:
        return {
            "blockchain": x.get("blockchain"),
            "symbol": x.get("symbol"),
            "amount": x.get("amount"),
            "amount_usd": x.get("amount_usd"),
            "from": x.get("from", {}).get("address") or x.get("from", {}).get("owner"),
            "to": x.get("to", {}).get("address") or x.get("to", {}).get("owner"),
            "timestamp": x.get("timestamp"),
            "hash": x.get("hash"),
            "source": "whale-alert",
        }

    @staticmethod
    def _normalize_etherscan_tx(x: dict[str, Any], usd_value: float) -> dict[str, Any]:
        return {
            "blockchain": "ethereum",
            "symbol": "ETH",
            "amount": int(x.get("value", "0")) / 1e18,
            "amount_usd": usd_value,
            "from": x.get("from"),
            "to": x.get("to"),
            "timestamp": x.get("timeStamp"),
            "hash": x.get("hash"),
            "source": "etherscan",
        }

    @staticmethod
    def _mock_whales(coin: str, min_usd: int) -> list[dict[str, Any]]:
        now = int(datetime.now(timezone.utc).timestamp())
        return [
            {"blockchain": coin, "symbol": coin[:3].upper(), "amount": 1250, "amount_usd": min_usd * 1.8, "from": "unknown_wallet", "to": "cold_wallet", "timestamp": now, "hash": "mock_accumulation_1", "source": "mock", "interpretation": "likely accumulation/outflow"},
            {"blockchain": coin, "symbol": coin[:3].upper(), "amount": 900, "amount_usd": min_usd * 1.2, "from": "exchange", "to": "unknown_wallet", "timestamp": now - 3600, "hash": "mock_outflow_2", "source": "mock", "interpretation": "exchange outflow"},
            {"blockchain": coin, "symbol": coin[:3].upper(), "amount": 700, "amount_usd": min_usd * 0.95, "from": "unknown_wallet", "to": "exchange", "timestamp": now - 7200, "hash": "mock_inflow_3", "source": "mock", "interpretation": "below threshold / watch"},
        ]

    @staticmethod
    def _bias_from_counts(buys: int, sells: int) -> str:
        if buys > sells:
            return "accumulation"
        if sells > buys:
            return "distribution"
        return "neutral"

    @staticmethod
    def _coingecko_id(coin: str) -> str:
        c = coin.lower()
        return CONFIG.get("coins", {}).get(c, {}).get("coingecko_id", c)

    @staticmethod
    def _symbol(coin: str) -> str:
        c = coin.lower()
        return CONFIG.get("coins", {}).get(c, {}).get("symbol", coin[:3].upper())

    @staticmethod
    def _mock_price(coin: str) -> float:
        base = {"bitcoin": 100000, "btc": 100000, "ethereum": 3000, "eth": 3000, "solana": 150, "dogecoin": 0.15}.get(coin.lower(), 100)
        return float(base)
