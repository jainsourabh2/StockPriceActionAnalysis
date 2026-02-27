"""
Reference script for placing orders through AngelOne SmartAPI.

╔══════════════════════════════════════════════════════════════╗
║  WARNING: This script can place REAL orders with real money. ║
║  DRY_RUN is True by default — it only PRINTS what it would  ║
║  do. Do NOT set DRY_RUN = False unless you fully understand  ║
║  what you're doing and have tested thoroughly in small qty.  ║
╚══════════════════════════════════════════════════════════════╝

Usage (dry run — safe, no real orders):
    python place_order.py

To enable real orders, change DRY_RUN to False in the code.
Do this at your own risk.
"""

import sys
from connect import get_session

# ─────────────────────────────────────────────────────────
# SAFETY FLAG — When True, no real orders are placed.
# Only prints what it WOULD do.
DRY_RUN = True
# ─────────────────────────────────────────────────────────


def place_buy_order(
    smart_api,
    symbol: str = "RELIANCE-EQ",
    token: str = "2885",
    qty: int = 1,
    price: float = 0,
    order_type: str = "MARKET",
    dry_run: bool = True,
) -> str | None:
    """
    Place a buy order.

    Args:
        smart_api: Authenticated SmartConnect session
        symbol: Trading symbol (e.g., "RELIANCE-EQ" for equity)
        token: SmartAPI symbol token
        qty: Number of shares
        price: Limit price (0 for market orders)
        order_type: "MARKET" or "LIMIT"
        dry_run: If True, only prints the order details without placing it

    Returns:
        Order ID if placed, None if dry run or failed.
    """
    order_params = {
        "variety": "NORMAL",
        "tradingsymbol": symbol,
        "symboltoken": token,
        "transactiontype": "BUY",
        "exchange": "NSE",
        "ordertype": order_type,
        "producttype": "DELIVERY",  # CNC (delivery). Use "INTRADAY" for MIS.
        "duration": "DAY",
        "quantity": qty,
        "price": price if order_type == "LIMIT" else 0,
        "squareoff": 0,
        "stoploss": 0,
        "triggerprice": 0,
    }

    print(f"\n📋 Order details:")
    print(f"   Action:     BUY")
    print(f"   Symbol:     {symbol}")
    print(f"   Quantity:   {qty}")
    print(f"   Type:       {order_type}")
    if order_type == "LIMIT":
        print(f"   Price:      ₹{price:,.2f}")
    print(f"   Product:    DELIVERY (CNC)")

    if dry_run:
        print(f"\n   [DRY RUN] Order NOT placed. Set dry_run=False to execute.")
        return None

    try:
        response = smart_api.placeOrder(order_params)
        print(f"\n   Order placed! Order ID: {response}")
        return response
    except Exception as e:
        print(f"\n   Order failed: {e}")
        return None


def place_sell_order(
    smart_api,
    symbol: str = "RELIANCE-EQ",
    token: str = "2885",
    qty: int = 1,
    price: float = 0,
    order_type: str = "MARKET",
    dry_run: bool = True,
) -> str | None:
    """
    Place a sell order. Same as buy but with transactiontype=SELL.
    """
    order_params = {
        "variety": "NORMAL",
        "tradingsymbol": symbol,
        "symboltoken": token,
        "transactiontype": "SELL",
        "exchange": "NSE",
        "ordertype": order_type,
        "producttype": "DELIVERY",
        "duration": "DAY",
        "quantity": qty,
        "price": price if order_type == "LIMIT" else 0,
        "squareoff": 0,
        "stoploss": 0,
        "triggerprice": 0,
    }

    print(f"\n📋 Order details:")
    print(f"   Action:     SELL")
    print(f"   Symbol:     {symbol}")
    print(f"   Quantity:   {qty}")
    print(f"   Type:       {order_type}")
    if order_type == "LIMIT":
        print(f"   Price:      ₹{price:,.2f}")

    if dry_run:
        print(f"\n   [DRY RUN] Order NOT placed. Set dry_run=False to execute.")
        return None

    try:
        response = smart_api.placeOrder(order_params)
        print(f"\n   Order placed! Order ID: {response}")
        return response
    except Exception as e:
        print(f"\n   Order failed: {e}")
        return None


def place_stoploss_order(
    smart_api,
    symbol: str = "RELIANCE-EQ",
    token: str = "2885",
    qty: int = 1,
    price: float = 0,
    trigger_price: float = 0,
    dry_run: bool = True,
) -> str | None:
    """
    Place a stop-loss limit order.

    The order activates when price hits trigger_price, then places a limit order at price.
    For a sell SL: trigger_price < current price, price <= trigger_price.
    """
    if trigger_price <= 0 or price <= 0:
        print("Stop-loss orders need both price and trigger_price > 0")
        return None

    order_params = {
        "variety": "STOPLOSS",
        "tradingsymbol": symbol,
        "symboltoken": token,
        "transactiontype": "SELL",
        "exchange": "NSE",
        "ordertype": "STOPLOSS_LIMIT",
        "producttype": "DELIVERY",
        "duration": "DAY",
        "quantity": qty,
        "price": price,
        "squareoff": 0,
        "stoploss": 0,
        "triggerprice": trigger_price,
    }

    print(f"\n📋 Stop-Loss Order:")
    print(f"   Symbol:         {symbol}")
    print(f"   Quantity:       {qty}")
    print(f"   Trigger Price:  ₹{trigger_price:,.2f}")
    print(f"   Limit Price:    ₹{price:,.2f}")

    if dry_run:
        print(f"\n   [DRY RUN] Order NOT placed.")
        return None

    try:
        response = smart_api.placeOrder(order_params)
        print(f"\n   Stop-loss order placed! Order ID: {response}")
        return response
    except Exception as e:
        print(f"\n   Order failed: {e}")
        return None


def check_order_status(smart_api, order_id: str, dry_run: bool = True) -> dict | None:
    """
    Check the status of an order by order ID.
    """
    if dry_run:
        print(f"\n   [DRY RUN] Would check status of order: {order_id}")
        return None

    try:
        order_book = smart_api.orderBook()

        if order_book and order_book.get("data"):
            for order in order_book["data"]:
                if order.get("orderid") == order_id:
                    print(f"\n   Order {order_id}:")
                    print(f"   Status:  {order.get('orderstatus')}")
                    print(f"   Symbol:  {order.get('tradingsymbol')}")
                    print(f"   Qty:     {order.get('quantity')}")
                    print(f"   Price:   {order.get('price')}")
                    return order

        print(f"   Order {order_id} not found in order book.")
        return None
    except Exception as e:
        print(f"   Failed to fetch order status: {e}")
        return None


def cancel_order(smart_api, order_id: str, variety: str = "NORMAL", dry_run: bool = True) -> bool:
    """
    Cancel an open order.

    Args:
        smart_api: Authenticated session
        order_id: The order ID to cancel
        variety: "NORMAL" for regular orders, "STOPLOSS" for SL orders
        dry_run: Safety flag
    """
    print(f"\n   Cancelling order: {order_id} (variety: {variety})")

    if dry_run:
        print(f"   [DRY RUN] Order NOT cancelled.")
        return False

    try:
        response = smart_api.cancelOrder(order_id, variety)
        print(f"   Order cancelled: {response}")
        return True
    except Exception as e:
        print(f"   Cancel failed: {e}")
        return False


def main():
    """
    Demo: walks through each operation in dry-run mode.
    Nothing is actually executed unless you change DRY_RUN.
    """
    if DRY_RUN:
        print("=" * 55)
        print("  RUNNING IN DRY-RUN MODE — no real orders will be placed")
        print("=" * 55)
    else:
        print("!" * 55)
        print("  LIVE MODE — orders WILL be placed with real money!")
        print("  Press Ctrl+C within 5 seconds to abort...")
        print("!" * 55)
        import time
        time.sleep(5)

    smart_api = get_session()

    # Example 1: Market buy order
    print("\n--- Example 1: Market Buy Order ---")
    order_id = place_buy_order(
        smart_api,
        symbol="RELIANCE-EQ",
        token="2885",
        qty=1,
        order_type="MARKET",
        dry_run=DRY_RUN,
    )

    # Example 2: Limit sell order
    print("\n--- Example 2: Limit Sell Order ---")
    place_sell_order(
        smart_api,
        symbol="RELIANCE-EQ",
        token="2885",
        qty=1,
        price=1350.00,
        order_type="LIMIT",
        dry_run=DRY_RUN,
    )

    # Example 3: Stop-loss order
    print("\n--- Example 3: Stop-Loss Order ---")
    place_stoploss_order(
        smart_api,
        symbol="RELIANCE-EQ",
        token="2885",
        qty=1,
        price=1245.00,
        trigger_price=1250.00,
        dry_run=DRY_RUN,
    )

    # Example 4: Check order status (only works with a real order ID)
    print("\n--- Example 4: Check Order Status ---")
    if order_id:
        check_order_status(smart_api, order_id, dry_run=DRY_RUN)
    else:
        print("   No order ID to check (dry run)")

    # Example 5: Cancel an order
    print("\n--- Example 5: Cancel Order ---")
    if order_id:
        cancel_order(smart_api, order_id, dry_run=DRY_RUN)
    else:
        print("   No order ID to cancel (dry run)")

    print("\nDone. All examples ran in", "DRY RUN" if DRY_RUN else "LIVE", "mode.")


if __name__ == "__main__":
    main()
