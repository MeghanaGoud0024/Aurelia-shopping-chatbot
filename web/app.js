/* ==========================================================================
   Aurelia AI Shopping Assistant - client

   No framework and no build step: the reviewer installs Python dependencies
   and opens a browser.

   Three principles:
   1. Structured data drives the UI, never parsed prose. Products, orders,
      cart, quote and dashboard figures arrive as typed payloads; a card is
      never reconstructed from what the model wrote.
   2. All server text becomes text nodes. The only markup generated here is
      our own - see renderMarkdown.
   3. Accessibility is structural: the rail is a real tablist with roving
      tabindex, the transcript is a live region, and focus is managed across
      async updates.
   ========================================================================== */

(() => {
  "use strict";

  const API = {
    session:   "/api/session",
    dashboard: "/api/dashboard",
    chat:      "/api/chat",
    reset:     "/api/chat/reset",
    cart:      "/api/cart",
    cartLine:  (id) => `/api/cart/${id}`,
    orders:    "/api/orders",
    order:     (n) => `/api/orders/${encodeURIComponent(n)}`,
    confirm:   "/api/checkout/confirm",
    feedback:  "/api/feedback",
    metrics:   "/api/ops/metrics",
  };

  const $ = (id) => document.getElementById(id);

  const dom = {
    transcript: $("transcript"),
    intro:      $("chat-intro"),
    composer:   $("composer"),
    input:      $("composer-input"),
    send:       $("send-button"),
    status:     $("composer-status"),
    modelName:  $("model-name"),
    avatar:     $("rail-avatar"),
    bagBadge:   $("bag-badge"),
    themeBtn:   $("theme-toggle"),
    resetBtn:   $("reset-chat"),
    main:       $("main"),
    navItems:   Array.from(document.querySelectorAll(".rail__item[data-view]")),
  };

  const state = { busy: false, sessionId: null, dashboard: null, lastTurn: null, spendRange: 6 };

  /* ------------------------------------------------------------- helpers */

  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  };

  const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); };

  const plural = (n, one, many) =>
    `${Number(n).toLocaleString()} ${Number(n) === 1 ? one : (many || one + "s")}`;

  const stars = (rating) =>
    "★".repeat(Math.round(rating)) + "☆".repeat(Math.max(0, 5 - Math.round(rating)));

  //: Chart keys that have already played their grow-in animation this
  //: session. Keyed by name rather than by node, because the spend bars are
  //: full-rebuilt (clear + recreate) on every dashboard refresh and so never
  //: carry state of their own across renders; the hero dial and meter persist
  //: as the same node and would work either way.
  const animatedCharts = new Set();

  /** Grow a chart element in from zero the first time it is drawn.
   *
   *  The dashboard reloads after every chat turn, so these elements are
   *  redrawn repeatedly with values that are usually unchanged. Forcing a
   *  zero start on every redraw made the chart visibly snap back to empty and
   *  re-grow on each message, which reads as broken rather than live.
   *
   *  `key` identifies the chart (not the node) so the "already animated"
   *  state survives the spend bars being torn down and rebuilt. On the first
   *  call for a key, the start value is written, a reflow is forced so the
   *  browser has a frame to transition *from*, then the target is written -
   *  doing this inside requestAnimationFrame instead would leave the element
   *  stuck at the start value if that frame were ever dropped. Every later
   *  call for the same key sets the target directly with no reset, so a
   *  changed value still transitions smoothly (the CSS transition on the
   *  property handles that), but an unchanged one produces no visible motion.
   */
  const growIn = (key, node, property, from, to) => {
    if (animatedCharts.has(key)) {
      node.style[property] = to;
      return;
    }
    node.style[property] = from;
    void node.offsetHeight;            // flush styles so the change transitions
    node.style[property] = to;
    animatedCharts.add(key);
  };

  const svgEl = (tag, attrs) => {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
    return node;
  };

  async function request(url, options = {}) {
    const response = await fetch(url, {
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!response.ok) {
      let detail = `Request failed (${response.status})`;
      try {
        const body = await response.json();
        detail = body?.detail?.error || body?.detail || body?.error || detail;
      } catch { /* non-JSON body; keep the status message */ }
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return response.status === 204 ? null : response.json();
  }

  /** Render a safe subset of markdown: emphasis, inline code, lists. Never HTML. */
  function renderMarkdown(text) {
    const fragment = document.createDocumentFragment();
    for (const block of String(text || "").split(/\n{2,}/)) {
      const lines = block.split("\n").filter((l) => l.trim());
      if (!lines.length) continue;
      const bulleted = lines.every((l) => /^\s*[-*]\s+/.test(l));
      const numbered = lines.every((l) => /^\s*\d+[.)]\s+/.test(l));
      if (bulleted || numbered) {
        const list = el(numbered ? "ol" : "ul");
        for (const line of lines) {
          const item = el("li");
          inline(item, line.replace(/^\s*(?:[-*]|\d+[.)])\s+/, ""));
          list.appendChild(item);
        }
        fragment.appendChild(list);
      } else {
        const p = el("p");
        lines.forEach((line, i) => { if (i) p.appendChild(el("br")); inline(p, line); });
        fragment.appendChild(p);
      }
    }
    return fragment;
  }

  function inline(parent, text) {
    const pattern = /(\*\*[^*]+\*\*|`[^`]+`)/g;
    let cursor = 0, match;
    while ((match = pattern.exec(text)) !== null) {
      if (match.index > cursor) parent.appendChild(document.createTextNode(text.slice(cursor, match.index)));
      const token = match[0];
      parent.appendChild(
        token.startsWith("**") ? el("strong", null, token.slice(2, -2)) : el("code", null, token.slice(1, -1))
      );
      cursor = pattern.lastIndex;
    }
    if (cursor < text.length) parent.appendChild(document.createTextNode(text.slice(cursor)));
  }

  const STATUS_TONE = {
    delivered: "good", shipped: "info", out_for_delivery: "info",
    confirmed: "", packed: "", pending_payment: "warn",
    return_requested: "warn", returned: "bad", cancelled: "bad",
  };
  const statusChip = (status, label) => {
    const tone = STATUS_TONE[status];
    return el("span", `chip${tone ? " chip--" + tone : ""}`, label || status);
  };

  /* -------------------------------------------------------------- rail */

  function selectView(name, { focus = false } = {}) {
    for (const item of dom.navItems) {
      const active = item.dataset.view === name;
      item.classList.toggle("is-active", active);
      item.setAttribute("aria-selected", String(active));
      item.tabIndex = active ? 0 : -1;
      const panel = $(item.getAttribute("aria-controls"));
      if (panel) panel.hidden = !active;
      if (active && focus) item.focus();
    }
    dom.main.scrollTop = 0;
    if (name === "orders") renderOrders();
    if (name === "bag") renderBagView();
    if (name === "governance") renderGovernance();
  }

  dom.navItems.forEach((item, index) => {
    item.addEventListener("click", () => selectView(item.dataset.view));
    item.addEventListener("keydown", (event) => {
      const step = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 }[event.key];
      let next = null;
      if (step) next = dom.navItems[(index + step + dom.navItems.length) % dom.navItems.length];
      else if (event.key === "Home") next = dom.navItems[0];
      else if (event.key === "End") next = dom.navItems[dom.navItems.length - 1];
      if (next) { event.preventDefault(); selectView(next.dataset.view, { focus: true }); }
    });
  });

  document.addEventListener("click", (event) => {
    const jump = event.target.closest("[data-view-jump]");
    if (jump) selectView(jump.dataset.viewJump);
    const prompt = event.target.closest("[data-prompt]");
    if (prompt) send(prompt.dataset.prompt);
  });

  /* --------------------------------------------------------- dashboard */

  async function loadDashboard() {
    try {
      state.dashboard = await request(API.dashboard);
    } catch {
      return;
    }
    const d = state.dashboard;

    $("home-date").textContent = d.today_readable;
    $("home-title").textContent = `${d.greeting}, ${d.customer.first_name}`;
    dom.avatar.textContent = d.customer.initials;
    dom.avatar.title = `${d.customer.name} (${d.customer.public_id}, ${d.customer.tier} tier). Only this customer's orders are visible.`;

    // Hero
    $("hero-eyebrow").textContent = d.hero.eyebrow;
    $("hero-title").textContent = d.hero.title;
    $("hero-caption").textContent = d.hero.caption;
    $("hero-value").textContent = d.hero.value_label;
    $("hero-target").textContent = d.hero.target_label;
    $("hero-pct").textContent = `${d.hero.progress_pct}%`;
    const dialCircumference = 2 * Math.PI * 50;
    growIn("hero-meter", $("hero-meter"), "width", "0%", `${d.hero.progress_pct}%`);
    growIn(
      "hero-dial", $("hero-dial"), "strokeDashoffset",
      String(dialCircumference),
      String(dialCircumference - (dialCircumference * d.hero.progress_pct) / 100),
    );

    // Stat tiles
    const tiles = $("stat-tiles");
    clear(tiles);
    for (const stat of d.stats) {
      const tile = el("div", "tile");
      tile.appendChild(el("div", "tile__value", stat.value));
      tile.appendChild(el("div", "tile__label", stat.label));
      tiles.appendChild(tile);
    }

    renderDonut(d.status_breakdown);
    renderSpend(d.spend, state.spendRange);
    renderNextDelivery(d.next_delivery);
    renderRecent(d.recent_orders);
    updateBagBadge(Number(d.stats.find((s) => s.key === "bag")?.value || 0));
  }

  const DONUT_COLORS = {
    delivered: "var(--good)", shipped: "var(--info)", out_for_delivery: "var(--accent-2)",
    packed: "var(--ink-3)", confirmed: "var(--accent)", pending_payment: "var(--warn)",
    return_requested: "var(--warn)", returned: "var(--ink-4)", cancelled: "var(--bad)",
  };

  function renderDonut(breakdown) {
    const svg = $("status-donut");
    const legend = $("status-legend");
    clear(svg); clear(legend);

    const total = breakdown.reduce((sum, b) => sum + b.count, 0);
    $("donut-total").textContent = total;
    if (!total) return;

    const radius = 46;
    const circumference = 2 * Math.PI * radius;
    let offset = 0;

    svg.appendChild(svgEl("circle", {
      cx: 60, cy: 60, r: radius, class: "donut__seg", stroke: "var(--sunk)",
      "stroke-dasharray": `${circumference} 0`,
    }));

    // The donut draws every status; the legend shows the top few and folds the
    // rest into one row, because nine legend rows is taller than the chart.
    const LEGEND_LIMIT = 5;
    let folded = 0;

    breakdown.forEach((slice, index) => {
      const length = (slice.count / total) * circumference;
      // A 2px gap between segments reads as separation without a border colour
      // that would have to be themed twice.
      const seg = svgEl("circle", {
        cx: 60, cy: 60, r: radius, class: "donut__seg",
        stroke: DONUT_COLORS[slice.status] || "var(--ink-3)",
        "stroke-dasharray": `${Math.max(0, length - 2)} ${circumference - Math.max(0, length - 2)}`,
        "stroke-dashoffset": String(-offset),
      });
      svg.appendChild(seg);
      offset += length;

      if (index >= LEGEND_LIMIT) { folded += slice.count; return; }

      const row = el("li");
      const dot = el("span", "legend__dot");
      dot.style.background = DONUT_COLORS[slice.status] || "var(--ink-3)";
      row.appendChild(dot);
      row.appendChild(el("span", null, slice.label));
      row.appendChild(el("span", "legend__count", slice.count));
      legend.appendChild(row);
    });

    if (folded) {
      const row = el("li", "legend__more");
      row.appendChild(el("span", null,
        `+ ${plural(breakdown.length - LEGEND_LIMIT, "other status", "other statuses")}`));
      row.appendChild(el("span", "legend__count", folded));
      legend.appendChild(row);
    }
  }

  function renderSpend(spend, months) {
    const host = $("spend-bars");
    clear(host);
    const points = spend.points.slice(-months);
    const peak = Math.max(...points.map((p) => p.amount_cents), 1);

    for (const point of points) {
      const bar = el("div", "bar");
      const pct = Math.round((point.amount_cents / peak) * 100);
      if (pct === 100) bar.classList.add("is-peak");
      const track = el("div", "bar__track");
      const fill = el("div", "bar__fill");
      fill.title = `${point.label}: ${point.display}`;
      track.appendChild(fill);
      bar.appendChild(track);
      bar.appendChild(el("div", "bar__label", point.label));
      host.appendChild(bar);
      growIn(`spend-bar-${point.label}`, fill, "height", "0%", `${Math.max(pct, 3)}%`);
    }
    $("spend-total").textContent = spend.twelve_month_total.display;
  }

  function renderNextDelivery(next) {
    const host = $("next-body");
    clear(host);
    if (!next) {
      host.appendChild(el("p", "empty", "Nothing in transit right now."));
      return;
    }
    host.appendChild(el("div", "next-card__day", next.date_readable));
    host.appendChild(el("div", "next-card__meta",
      `Order ${next.order_number} · ${next.carrier || "carrier pending"}`));
    host.appendChild(el("div", "next-card__msg", next.message));

    const chip = statusChip(next.status, next.status_label);
    const wrap = el("div");
    wrap.appendChild(chip);
    host.appendChild(wrap);

    // A small route flourish, matching the tile in the reference layout.
    const route = el("div", "next-card__route");
    const svg = svgEl("svg", { viewBox: "0 0 120 40", "aria-hidden": "true" });
    svg.appendChild(svgEl("path", {
      d: "M4 30 C 22 30, 24 10, 42 10 S 70 30, 88 24 S 110 8, 116 12",
      fill: "none", stroke: "currentColor", "stroke-width": "2.4", "stroke-linecap": "round",
    }));
    svg.appendChild(svgEl("circle", { cx: "4", cy: "30", r: "3.4", fill: "currentColor", stroke: "none" }));
    svg.appendChild(svgEl("circle", { cx: "116", cy: "12", r: "3.4", fill: "currentColor", stroke: "none" }));
    route.appendChild(svg);
    host.appendChild(route);
  }

  function renderRecent(orders) {
    const host = $("recent-list");
    clear(host);
    if (!orders.length) {
      host.appendChild(el("p", "empty", "No orders on this account yet."));
      return;
    }
    for (const order of orders) {
      const row = el("div", "row");
      const main = el("div", "row__main");
      main.appendChild(el("div", "row__title", `Order ${order.order_number}`));
      main.appendChild(el("div", "row__meta",
        `${order.placed_readable} · ${plural(order.item_count, "item")}${order.first_item ? " · " + order.first_item : ""}`));
      row.appendChild(main);
      row.appendChild(statusChip(order.status, order.status_label));
      row.appendChild(el("div", "row__value", order.total.display));
      host.appendChild(row);
    }
  }

  document.querySelectorAll(".segmented__btn").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".segmented__btn").forEach((b) => b.classList.remove("is-active"));
      button.classList.add("is-active");
      state.spendRange = Number(button.dataset.range);
      if (state.dashboard) renderSpend(state.dashboard.spend, state.spendRange);
    });
  });

  /* ------------------------------------------------------------- orders */

  async function renderOrders() {
    const host = $("orders-body");
    clear(host);
    host.appendChild(el("p", "empty", "Loading your orders..."));
    try {
      const listing = await request(`${API.orders}?limit=20`);
      clear(host);
      if (!listing.orders.length) {
        host.appendChild(el("p", "empty", listing.note || "No orders on this account."));
        return;
      }
      for (const summary of listing.orders) {
        host.appendChild(orderSummaryCard(summary));
      }
    } catch (error) {
      clear(host);
      host.appendChild(el("p", "empty", `Could not load orders. ${error.message}`));
    }
  }

  /** Collapsed order card. Detail and the shipment timeline are one request
   *  away, so the list stays scannable instead of paying full card height for
   *  every order the customer has ever placed. */
  function orderSummaryCard(summary) {
    const card = el("article", "order order--summary");

    const head = el("div", "order__head");
    const num = el("div", "order__num");
    num.appendChild(document.createTextNode(`Order ${summary.order_number} `));
    num.appendChild(el("span", null, `· ${formatDate(summary.placed_at)}`));
    head.appendChild(num);
    head.appendChild(statusChip(summary.status, summary.status_label));
    card.appendChild(head);

    const body = el("div", "order__body");
    const row = el("div", "order__summary-row");

    const meta = el("div");
    meta.appendChild(el("div", "row__meta", plural(summary.item_count, "item")));
    const when = summary.delivered_at
      ? `Delivered ${formatDate(summary.delivered_at)}`
      : summary.estimated_delivery_at
        ? `Estimated ${formatDate(summary.estimated_delivery_at)}`
        : "";
    if (when) meta.appendChild(el("div", "row__meta", when));
    row.appendChild(meta);
    row.appendChild(el("div", "order__summary-total", summary.total.display));
    body.appendChild(row);

    const more = el("button", "linkish");
    more.type = "button";
    more.textContent = "Show detail and timeline";
    more.addEventListener("click", async () => {
      more.disabled = true;
      more.textContent = "Loading...";
      try {
        card.replaceWith(orderCard(await request(API.order(summary.order_number))));
      } catch (error) {
        more.disabled = false;
        more.textContent = `Could not load. ${error.message}`;
      }
    });
    body.appendChild(more);
    card.appendChild(body);
    return card;
  }

  function orderCard(order) {
    const card = el("article", "order");

    const head = el("div", "order__head");
    const num = el("div", "order__num");
    num.appendChild(document.createTextNode(`Order ${order.order_number} `));
    num.appendChild(el("span", null, `· ${formatDate(order.placed_at)}`));
    head.appendChild(num);
    head.appendChild(statusChip(order.status, order.status_label));
    card.appendChild(head);

    const body = el("div", "order__body");
    if (order.delivery_message) body.appendChild(el("p", "order__delivery", order.delivery_message));

    if (order.items?.length) {
      const lines = el("div", "order__lines");
      for (const item of order.items) {
        const line = el("div", "order__line");
        const left = el("div");
        left.appendChild(el("b", null, `${item.quantity} x ${item.product_name}`));
        left.appendChild(el("div", "order__line-meta", `${item.size} · ${item.color} · ${item.brand}`));
        line.appendChild(left);
        line.appendChild(el("div", null, item.line_total.display));
        lines.appendChild(line);
      }
      body.appendChild(lines);
    }

    if (order.total) {
      const totals = el("div", "order__totals");
      totals.appendChild(el("span", null, "Order total"));
      totals.appendChild(el("span", null, order.total.display));
      body.appendChild(totals);
    }
    if (order.tracking_number) {
      body.appendChild(el("div", "order__line-meta",
        `Tracking ${order.tracking_number} with ${order.carrier}`));
    }
    if (order.timeline?.length) {
      const list = el("ul", "timeline");
      for (const event of order.timeline) {
        const item = el("li");
        item.appendChild(el("b", null, event.label));
        item.appendChild(document.createTextNode(
          `${formatDate(event.occurred_at)}${event.location ? " · " + event.location : ""}`));
        list.appendChild(item);
      }
      body.appendChild(list);
    }
    card.appendChild(body);
    return card;
  }

  const formatDate = (iso) => {
    if (!iso) return "";
    const date = new Date(iso);
    return Number.isNaN(date.getTime())
      ? "" : date.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
  };

  /* ---------------------------------------------------------------- bag */

  function updateBagBadge(count) {
    dom.bagBadge.hidden = !count;
    dom.bagBadge.textContent = count;
  }

  async function renderBagView(cart) {
    const host = $("bag-body");
    try {
      const data = cart || await request(API.cart);
      clear(host);
      updateBagBadge(data.item_count);

      if (!data.lines.length) {
        host.appendChild(el("p", "empty",
          "Your bag is empty. Ask the assistant to add something and it will appear here."));
        return;
      }

      const list = el("div", "card list-card");
      for (const line of data.lines) {
        const row = el("div", "row");
        const main = el("div", "row__main");
        main.appendChild(el("div", "row__title", line.product_name));
        main.appendChild(el("div", "row__meta",
          `${line.quantity} x ${line.size} · ${line.color} · ${line.unit_price.display}`));
        row.appendChild(main);
        row.appendChild(el("div", "row__value", line.line_total.display));

        const remove = el("button", "row__action", "Remove");
        remove.type = "button";
        remove.setAttribute("aria-label", `Remove ${line.product_name} from your bag`);
        remove.addEventListener("click", async () => {
          remove.disabled = true;
          try {
            const updated = await request(API.cartLine(line.variant_id), { method: "DELETE" });
            renderBagView(updated);
            loadDashboard();
            announce(`${line.product_name} removed.`);
          } catch (error) { remove.disabled = false; announce(error.message); }
        });
        row.appendChild(remove);
        list.appendChild(row);
      }
      host.appendChild(list);

      const totals = el("div", "card");
      for (const [label, value] of [
        ["Subtotal", data.subtotal.display],
        ["Shipping", data.shipping.amount_cents === 0 ? "Free" : data.shipping.display],
        ["Tax", data.tax.display],
      ]) {
        const row = el("div", "order__line");
        row.appendChild(el("span", null, label));
        row.appendChild(el("span", null, value));
        totals.appendChild(row);
      }
      const grand = el("div", "order__totals");
      grand.appendChild(el("span", null, "Total"));
      grand.appendChild(el("span", null, data.total.display));
      totals.appendChild(grand);
      if (data.note) totals.appendChild(el("p", "order__delivery", data.note));
      host.appendChild(totals);

      const checkout = el("button", "button button--primary", "Ask the assistant to check out");
      checkout.type = "button";
      checkout.addEventListener("click", () => send("I'd like to check out"));
      host.appendChild(checkout);
    } catch (error) {
      clear(host);
      host.appendChild(el("p", "empty", `Could not load your bag. ${error.message}`));
    }
  }

  async function refreshCart() {
    try {
      const cart = await request(API.cart);
      updateBagBadge(cart.item_count);
      if (!$("view-bag").hidden) renderBagView(cart);
    } catch { /* keep the last good state */ }
  }

  /* -------------------------------------------------------------- trace */

  function traceList(trace) {
    const list = el("ol", "trace");
    for (const step of trace) {
      const item = el("li", "trace__step");
      item.dataset.kind = step.kind;
      item.dataset.status = step.status;
      item.appendChild(el("span", "trace__index", step.step));

      const middle = el("div");
      const label = el("div", "trace__label");
      if (step.tool_name) {
        label.appendChild(el("code", null, step.tool_name));
        if (step.detail === "mutating call") {
          label.appendChild(document.createTextNode(" "));
          label.appendChild(el("span", "chip", "writes"));
        }
      } else {
        label.textContent = step.label;
      }
      middle.appendChild(label);

      if (step.arguments && Object.keys(step.arguments).length) {
        middle.appendChild(el("div", "trace__args", JSON.stringify(step.arguments)));
      }
      const detail = step.result_summary || (step.detail !== "mutating call" ? step.detail : "");
      if (detail) middle.appendChild(el("div", "trace__detail", detail));
      item.appendChild(middle);
      item.appendChild(el("span", "trace__latency", step.latency_ms ? `${step.latency_ms}ms` : ""));
      list.appendChild(item);
    }
    return list;
  }

  function traceSummary(payload) {
    const summary = el("div", "trace__summary");
    const calls = payload.trace.filter((s) => s.kind === "tool_call").length;
    summary.appendChild(el("span", null, plural(calls, "backend call")));
    summary.appendChild(el("span", null, `${payload.latency_ms}ms`));
    summary.appendChild(el("span", null, payload.grounded ? "Grounded" : "Ungrounded, answer withheld"));
    summary.appendChild(el("span", null, payload.model));
    return summary;
  }

  function renderTraceView(payload) {
    const host = $("trace-body");
    clear(host);
    if (!payload) {
      host.appendChild(el("p", "empty", "Ask the assistant something to see how the answer was produced."));
      return;
    }
    const card = el("div", "card");
    card.appendChild(traceList(payload.trace));
    card.appendChild(traceSummary(payload));
    host.appendChild(card);
  }

  /* --------------------------------------------------------- governance */

  async function renderGovernance() {
    const host = $("governance-body");
    try {
      const metrics = await request(API.metrics);
      clear(host);

      const grid = el("div", "metric-grid");
      for (const [value, label] of [
        [metrics.tool_calls_total.toLocaleString(), "Backend calls"],
        [metrics.guardrail_events_total.toLocaleString(), "Guardrail checks"],
        [metrics.guardrail_blocks.toLocaleString(), "Blocked"],
        [`${(metrics.guardrail_block_rate * 100).toFixed(1)}%`, "Block rate"],
      ]) {
        const tile = el("div", "tile");
        tile.appendChild(el("div", "tile__value", value));
        tile.appendChild(el("div", "tile__label", label));
        grid.appendChild(tile);
      }
      host.appendChild(grid);

      if (metrics.tools.length) {
        const card = el("div", "card");
        card.appendChild(el("p", "card__label", "Tool usage"));
        const busiest = Math.max(...metrics.tools.map((t) => t.calls), 1);
        for (const tool of metrics.tools.slice(0, 10)) {
          const row = el("div", "bar-row");
          row.appendChild(el("span", "bar-row__name", tool.tool));
          const track = el("div", "bar-row__track");
          const fill = el("div", "bar-row__fill");
          fill.style.width = `${(tool.calls / busiest) * 100}%`;
          track.appendChild(fill);
          row.appendChild(track);
          row.appendChild(el("span", "bar-row__value", tool.calls));
          card.appendChild(row);
        }
        host.appendChild(card);
      }

      if (metrics.guardrails.length) {
        const card = el("div", "card");
        card.appendChild(el("p", "card__label", "Guardrail decisions"));
        for (const rule of metrics.guardrails.slice(0, 12)) {
          const row = el("div", "rule-row");
          row.appendChild(el("span", "rule-row__name", `${rule.stage}/${rule.rule}`));
          const action = el("span", "rule-row__action", `${rule.action} · ${rule.count}`);
          action.dataset.action = rule.action;
          row.appendChild(action);
          card.appendChild(row);
        }
        host.appendChild(card);
      }
    } catch (error) {
      clear(host);
      host.appendChild(el("p", "empty", `Could not load metrics. ${error.message}`));
    }
  }

  $("refresh-governance").addEventListener("click", renderGovernance);

  /* --------------------------------------------------------- chat cards */

  function productMini(product) {
    const card = el("article", "mini");
    const top = el("div", "mini__top");
    const head = el("div");
    head.appendChild(el("div", "mini__brand", product.brand));
    head.appendChild(el("div", "mini__name", product.name));
    top.appendChild(head);
    if (product.discount_pct > 0) top.appendChild(el("span", "chip", `-${product.discount_pct}%`));
    card.appendChild(top);

    const priceRow = el("div", "mini__row");
    const price = el("div", "mini__price", product.price.display);
    if (product.discount_pct > 0) price.appendChild(el("span", "mini__was", product.list_price.display));
    priceRow.appendChild(price);

    const stock = product.total_stock;
    priceRow.appendChild(
      !product.in_stock ? el("span", "chip chip--bad", "Out of stock")
      : stock <= 12 ? el("span", "chip chip--warn", `${stock} left`)
      : el("span", "chip chip--good", "In stock")
    );
    card.appendChild(priceRow);

    const rating = el("div", "mini__meta");
    rating.appendChild(el("span", "stars", stars(product.rating)));
    rating.appendChild(document.createTextNode(` ${product.rating} (${product.review_count.toLocaleString()})`));
    card.appendChild(rating);

    if (product.available_sizes?.length) {
      card.appendChild(el("div", "mini__meta", `Sizes ${product.available_sizes.join(", ")}`));
    }
    if (product.available_colors?.length) {
      card.appendChild(el("div", "mini__meta", `Colours ${product.available_colors.join(", ")}`));
    }
    return card;
  }

  /** The purchase confirmation card.
   *  The token is issued to the browser and withheld from the model, so this
   *  click is the only path to a charge. */
  function quoteCard(quote) {
    const card = el("section", "quote");
    card.setAttribute("aria-label", "Confirm your purchase");

    const head = el("div", "quote__head");
    head.appendChild(el("p", "quote__title", "Confirm your order"));
    head.appendChild(el("p", "quote__sub",
      `Nothing is charged until you confirm. ${plural(quote.lines.length, "item")} to ${quote.shipping_address}.`));
    card.appendChild(head);

    const body = el("div", "quote__body");
    for (const line of quote.lines) {
      const row = el("div", "quote__row");
      row.appendChild(el("span", null, `${line.quantity} x ${line.product_name} (${line.size}/${line.color})`));
      row.appendChild(el("span", null, line.line_total.display));
      body.appendChild(row);
    }
    for (const [label, value] of [
      ["Subtotal", quote.subtotal.display],
      ["Shipping", quote.shipping.amount_cents === 0 ? "Free" : quote.shipping.display],
      ["Tax", quote.tax.display],
    ]) {
      const row = el("div", "quote__row");
      row.appendChild(el("span", null, label));
      row.appendChild(el("span", null, value));
      body.appendChild(row);
    }
    const total = el("div", "quote__row quote__row--total");
    total.appendChild(el("span", null, "Total"));
    total.appendChild(el("span", null, quote.total.display));
    body.appendChild(total);
    for (const warning of quote.warnings || []) body.appendChild(el("p", "quote__warning", warning));
    card.appendChild(body);

    const actions = el("div", "quote__actions");
    const confirm = el("button", "button button--primary", `Confirm ${quote.total.display}`);
    confirm.type = "button";
    const cancel = el("button", "button button--quiet", "Not now");
    cancel.type = "button";

    confirm.addEventListener("click", async () => {
      confirm.disabled = cancel.disabled = true;
      confirm.textContent = "Placing your order...";
      try {
        const receipt = await request(API.confirm, {
          method: "POST",
          body: JSON.stringify({ confirmation_token: quote.confirmation_token }),
        });
        const done = el("div", "receipt");
        done.appendChild(el("p", "receipt__title", `Order ${receipt.order_number} confirmed`));
        done.appendChild(el("p", "receipt__body",
          `${receipt.total.display} by ${receipt.payment_method.replace(/_/g, " ")}. ` +
          `Estimated delivery ${receipt.estimated_delivery_readable}.`));
        card.replaceWith(done);
        announce(`Order ${receipt.order_number} confirmed.`);
        refreshCart();
        loadDashboard();
      } catch (error) {
        confirm.disabled = cancel.disabled = false;
        confirm.textContent = `Confirm ${quote.total.display}`;
        body.appendChild(el("p", "quote__warning", error.message));
        announce(`Could not place the order. ${error.message}`);
      }
    });
    cancel.addEventListener("click", () => {
      card.replaceWith(el("p", "mini__meta", "Purchase cancelled. Your bag is unchanged."));
    });

    actions.appendChild(confirm);
    actions.appendChild(cancel);
    card.appendChild(actions);
    return card;
  }

  /* ------------------------------------------------------------ messages */

  function announce(text) {
    dom.status.textContent = text;
    window.setTimeout(() => { if (dom.status.textContent === text) dom.status.textContent = ""; }, 6000);
  }

  function addUser(text) {
    const node = document.getElementById("tpl-message-user").content.cloneNode(true);
    node.querySelector(".msg__bubble").textContent = text;
    dom.transcript.appendChild(node);
    scrollChat();
  }

  function addTyping() {
    const node = document.getElementById("tpl-typing").content.cloneNode(true);
    const article = node.querySelector(".msg");
    dom.transcript.appendChild(node);
    scrollChat();
    const notes = ["Checking our systems", "Looking that up", "Reading the results"];
    let index = 0;
    const noteNode = article.querySelector(".dots__note");
    const timer = window.setInterval(() => {
      index = (index + 1) % notes.length;
      if (noteNode) noteNode.textContent = notes[index];
    }, 2200);
    return () => { window.clearInterval(timer); article.remove(); };
  }

  function addAssistant(payload) {
    const node = document.getElementById("tpl-message-assistant").content.cloneNode(true);
    const article = node.querySelector(".msg");
    const bubble = node.querySelector(".msg__bubble");
    const cards = node.querySelector(".msg__cards");
    const citations = node.querySelector(".msg__citations");
    const toggle = node.querySelector(".msg__trace-toggle");
    const pane = node.querySelector(".msg__trace");

    bubble.appendChild(renderMarkdown(payload.reply));
    if (payload.blocked) article.classList.add("msg--blocked");
    if (!payload.grounded) article.classList.add("msg--ungrounded");

    payload.products?.forEach((p) => cards.appendChild(productMini(p)));
    payload.orders?.forEach((o) => cards.appendChild(orderCard(o)));
    if (payload.checkout_quote) cards.appendChild(quoteCard(payload.checkout_quote));

    if (payload.citations?.length) {
      citations.hidden = false;
      payload.citations.forEach((c) => citations.appendChild(el("span", "citation", c)));
    }

    pane.appendChild(traceList(payload.trace));
    pane.appendChild(traceSummary(payload));

    const label = toggle.querySelector(".trace-label");
    const calls = payload.trace.filter((s) => s.kind === "tool_call").length;
    const openLabel = calls ? `Show the ${plural(calls, "backend call")}` : "How this was answered";
    label.textContent = openLabel;
    toggle.addEventListener("click", () => {
      const open = toggle.getAttribute("aria-expanded") === "true";
      toggle.setAttribute("aria-expanded", String(!open));
      pane.hidden = open;
      label.textContent = open ? openLabel : "Hide the trace";
    });

    wireFeedback(node.querySelector(".msg__feedback"), payload.turn_id);
    dom.transcript.appendChild(node);
    scrollChat();
  }

  function wireFeedback(group, turnId) {
    const buttons = Array.from(group.querySelectorAll(".fb"));
    buttons.forEach((button) => {
      button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", async () => {
        const rating = button.dataset.rating;
        let reason = "";
        if (rating === "not_helpful") {
          reason = (window.prompt(
            "Thanks for flagging it. What went wrong? (wrong answer, missing information, too slow, other)"
          ) || "").slice(0, 60);
        }
        buttons.forEach((b) => { b.disabled = true; b.setAttribute("aria-pressed", String(b === button)); });
        group.classList.add("is-answered");
        try {
          await request(API.feedback, {
            method: "POST",
            body: JSON.stringify({ session_id: state.sessionId || "", turn_id: turnId, rating, reason, comment: "" }),
          });
          announce("Recorded against this exact turn.");
        } catch { announce("Could not record that feedback."); }
      });
    });
  }

  const scrollChat = () => requestAnimationFrame(() => {
    dom.transcript.scrollTop = dom.transcript.scrollHeight;
  });

  /* ---------------------------------------------------------------- send */

  async function send(text) {
    const message = String(text || "").trim();
    if (!message || state.busy) return;

    if (dom.intro) { dom.intro.remove(); dom.intro = null; }
    state.busy = true;
    dom.send.disabled = true;
    dom.input.value = "";
    resizeInput();
    dom.status.textContent = "";

    addUser(message);
    const removeTyping = addTyping();

    try {
      const payload = await request(API.chat, { method: "POST", body: JSON.stringify({ message }) });
      removeTyping();
      addAssistant(payload);
      state.lastTurn = payload;
      renderTraceView(payload);
      if (payload.cart) { updateBagBadge(payload.cart.item_count); if (!$("view-bag").hidden) renderBagView(payload.cart); }
      else refreshCart();
      loadDashboard();
    } catch (error) {
      removeTyping();
      addAssistant({
        reply: `I could not reach the assistant service. ${error.message}`,
        trace: [], products: [], orders: [], citations: [],
        turn_id: "error", latency_ms: 0, model: "-", grounded: true, blocked: true,
      });
    } finally {
      state.busy = false;
      dom.send.disabled = false;
      dom.input.focus();
    }
  }

  /* --------------------------------------------------------------- theme */

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    dom.themeBtn.setAttribute("aria-label", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
    try { localStorage.setItem("aurelia-theme", theme); } catch { /* private mode */ }
  }
  dom.themeBtn.addEventListener("click", () => {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  });

  /* ------------------------------------------------------------ composer */

  function resizeInput() {
    dom.input.style.height = "auto";
    dom.input.style.height = `${Math.min(dom.input.scrollHeight, 132)}px`;
  }
  dom.input.addEventListener("input", resizeInput);
  dom.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); send(dom.input.value); }
  });
  dom.composer.addEventListener("submit", (event) => { event.preventDefault(); send(dom.input.value); });

  dom.resetBtn.addEventListener("click", async () => {
    try { await request(API.reset, { method: "POST" }); } catch { /* still clear locally */ }
    clear(dom.transcript);
    state.lastTurn = null;
    renderTraceView(null);
    announce("Started a new conversation.");
    dom.input.focus();
  });

  /* ---------------------------------------------------------------- init */

  (async function init() {
    try {
      const stored = localStorage.getItem("aurelia-theme");
      applyTheme(stored || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
    } catch { applyTheme("light"); }

    try {
      const session = await request(API.session);
      state.sessionId = session.session_id;
      dom.modelName.textContent = session.model;
    } catch {
      dom.modelName.textContent = "offline";
    }

    await loadDashboard();
    renderTraceView(null);
    resizeInput();
    dom.input.focus();

    // Deep links: /#ask=<question> and ?theme=light|dark. Shareable worked
    // examples, and how the README screenshots are captured reproducibly.
    const params = new URLSearchParams(window.location.search);
    const themeParam = params.get("theme");
    if (themeParam === "light" || themeParam === "dark") applyTheme(themeParam);
    const view = params.get("view");
    if (view && dom.navItems.some((i) => i.dataset.view === view)) selectView(view);

    if (window.location.hash.startsWith("#ask=")) {
      const question = decodeURIComponent(window.location.hash.slice(5)).trim();
      if (question) send(question);
    }
  })();
})();
