---
title: Payments, Pricing and Promotions
topic: payments
version: 2.1
effective: 2026-04-01
owner: Aurelia Commerce
---

# Payments, Pricing and Promotions

## Accepted payment methods

Card (Visa, Mastercard, American Express), UPI, net banking, Aurelia Wallet, and cash on delivery. Cash on delivery is available on orders under 300.00 USD in supported postcodes only, and is not available on international orders.

## When you are charged

Placing an order creates a payment **authorisation**, which reserves the funds without moving them. The authorisation is captured when the order reaches `packed`. An order cancelled before that point never results in a charge; the authorisation is simply released, though the pending line may take 1 to 3 business days to clear from a bank statement.

## Pricing

All prices shown are in US dollars and include applicable GST or VAT where the destination market requires tax-inclusive display. Import duties on international orders are not included and are payable by the recipient.

The price recorded on an order is the price at the moment of checkout. A later price drop does not retroactively change a placed order, and Aurelia does not operate a price-match-after-purchase scheme.

## Promotions

- One promotion code per order. Codes do not stack.
- Promotion codes do not apply to items already marked Final Sale.
- Loyalty tier discounts apply automatically and do stack with a promotion code.
- If a promotional order is partially returned, the discount is recalculated across the retained items, which can reduce the refund below the line price paid.

## Loyalty tiers

| Tier | Qualifying spend (rolling 12 months) | Benefits |
| --- | --- | --- |
| Standard | Under 250 USD | Standard shipping rates |
| Silver | 250 to 749 USD | Free standard shipping |
| Gold | 750 to 1,999 USD | Free express shipping, 60-day returns |
| Platinum | 2,000 USD and above | Free priority shipping, 60-day returns, early access |

## Failed payments

If an authorisation fails, the order is held in `pending_payment` for 24 hours and the customer is prompted to retry. After 24 hours without a successful authorisation the order is cancelled automatically and any reserved stock is released.
