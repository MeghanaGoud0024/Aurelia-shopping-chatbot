---
title: Order Changes and Cancellation
topic: cancellation
version: 2.4
effective: 2026-02-10
owner: Aurelia Order Management
---

# Order Changes and Cancellation

## When an order can be cancelled

Cancellation is possible while the order is in `pending_payment`, `confirmed` or `packed`. These are the states where the parcel has not yet been handed to a carrier.

Once an order reaches `shipped`, it can no longer be cancelled, because the parcel is in the carrier network and Aurelia no longer physically controls it. The route at that point is to refuse delivery or to return the item after it arrives.

| Status | Cancellable | Notes |
| --- | --- | --- |
| pending_payment | Yes | Cancelled immediately, no charge was taken |
| confirmed | Yes | Refund initiated same day |
| packed | Yes | Warehouse pull-back, refund within 24 hours |
| shipped | No | Refuse delivery or return after arrival |
| out_for_delivery | No | Refuse delivery at the door |
| delivered | No | Use the returns process |
| cancelled / returned | No | Already terminal |

## Partial cancellation

Individual lines can be cancelled from a multi-item order while the order is in `confirmed`. Once picking begins the order is treated as a single unit and only a full cancellation is possible.

## Refund timing on cancellation

Cancellation before dispatch releases the payment authorisation rather than processing a refund, so funds typically reappear faster than a post-delivery return: within 1 to 3 business days on card, and immediately on wallet.

## Modifying an order

Quantity, size and colour cannot be edited after checkout. The item mix is locked at payment authorisation. To change any of these, cancel while the order is still cancellable and place a new order.
