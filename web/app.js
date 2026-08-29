/* ==========================================================================
   Aurelia AI Shopping Assistant - client

   No framework and no build step, which is a deliberate reading of the
   "runs on standard developer hardware with free tools" constraint: the
   reviewer clones, installs Python dependencies, and opens a browser.

   Three principles shape this file:

   1. Structured data drives the UI, never parsed prose. The server sends
      products, orders, cart and quote as typed payloads alongside the reply
      text, so a card is never reconstructed from what the model wrote.
   2. All server text is inserted as text nodes, never as HTML. The only
      markup we generate is our own. See `renderMarkdown`.
   3. Accessibility is structural, not decorative: the transcript is a live
      region, tabs implement the roving-tabindex pattern, focus is managed
      across async updates, and motion honours prefers-reduced-motion.
   ========================================================================== */

(() => {
  "use strict";

  const API = {
    session:    "/api/session",
    chat:       "/api/chat",
    cart:       "/api/cart",
    removeLine: (variantId) => `/api/cart/${variantId}`,
    confirm:    "/api/checkout/confirm",
    feedback:   "/api/feedback",
    metrics:    "/api/ops/metrics",
  };

  const dom = {
    transcript:     document.getElementById("transcript"),
    welcome:        document.getElementById("welcome"),
    composer:       document.getElementById("composer"),
    input:          document.getElementById("composer-input"),
    sendButton:     document.getElementById("send-button"),
    status:         document.getElementById("composer-status"),
    identityName:   document.getElementById("identity-name"),
    modelName:      document.getElementById("model-name"),
    themeToggle:    document.getElementById("theme-toggle"),
    cartBody:       document.getElementById("cart-body"),
    cartCount:      document.getElementById("cart-count"),
    traceBody:      document.getElementById("trace-body"),
    governanceBody: document.getElementById("governance-body"),
    refreshGov:     document.getElementById("refresh-governance"),
    tabs:           Array.from(document.querySelectorAll('[role="tab"]')),
  };

  const state = {
    busy: false,
    lastTrace: null,
    sessionId: null,
  };

  /* ------------------------------------------------------------- helpers */

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  };

  const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); };

  /** Render a safe subset of markdown. Emphasis and lists only, no raw HTML.
   *  Every literal from the server becomes a text node, so a reply containing
   *  angle brackets is displayed rather than executed. */
  function renderMarkdown(text) {
    const fragment = document.createDocumentFragment();
    const blocks = String(text || "").split(/\n{2,}/);

    for (const block of blocks) {
      const lines = block.split("\n").filter((l) => l.trim());
      if (!lines.length) continue;

      const bulleted = lines.every((l) => /^\s*[-*]\s+/.test(l));
      const numbered = lines.every((l) => /^\s*\d+[.)]\s+/.test(l));

      if (bulleted || numbered) {
        const list = el(numbered ? "ol" : "ul");
        for (const line of lines) {
          const item = el("li");
          appendInline(item, line.replace(/^\s*(?:[-*]|\d+[.)])\s+/, ""));
          list.appendChild(item);
        }
        fragment.appendChild(list);
      } else {
        const paragraph = el("p");
        lines.forEach((line, index) => {
          if (index) paragraph.appendChild(el("br"));
          appendInline(paragraph, line);
        });
        fragment.appendChild(paragraph);
      }
    }
    return fragment;
  }

  /** Split on **bold** and `code`, emitting text nodes for everything else. */
  function appendInline(parent, text) {
    const pattern = /(\*\*[^*]+\*\*|`[^`]+`)/g;
    let cursor = 0;
    let match;
    while ((match = pattern.exec(text)) !== null) {
      if (match.index > cursor) {
        parent.appendChild(document.createTextNode(text.slice(cursor, match.index)));
      }
      const token = match[0];
      if (token.startsWith("**")) {
        parent.appendChild(el("strong", null, token.slice(2, -2)));
      } else {
        parent.appendChild(el("code", null, token.slice(1, -1)));
      }
      cursor = pattern.lastIndex;
    }
    if (cursor < text.length) {
      parent.appendChild(document.createTextNode(text.slice(cursor)));
    }
  }

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
      } catch { /* non-JSON error body; keep the status message */ }
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return response.status === 204 ? null : response.json();
  }

  /** "1 call" / "3 calls". Writing "call(s)" everywhere reads as unfinished. */
  const plural = (count, singular, pluralForm) =>
    `${count.toLocaleString()} ${count === 1 ? singular : (pluralForm || singular + "s")}`;

  const stars = (rating) => {
    const filled = Math.round(rating);
    return "★".repeat(filled) + "☆".repeat(Math.max(0, 5 - filled));
  };

  const formatDate = (iso) => {
    if (!iso) return "";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "";
    return date.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
  };

  /* --------------------------------------------------------------- cards */

  function productCard(product) {
    const card = el("article", "product");

    const top = el("div", "product__top");
    const heading = el("div");
    heading.appendChild(el("div", "product__brand", product.brand));
    heading.appendChild(el("h3", "product__name", product.name));
    top.appendChild(heading);
    if (product.discount_pct > 0) {
      top.appendChild(el("span", "badge", `-${product.discount_pct}%`));
    }
    card.appendChild(top);

    const priceRow = el("div", "product__price-row");
    priceRow.appendChild(el("span", "product__price", product.price.display));
    if (product.discount_pct > 0) {
      priceRow.appendChild(el("span", "product__was", product.list_price.display));
    }
    card.appendChild(priceRow);

    const meta = el("div", "product__meta");
    const rating = el("span", "product__rating");
    rating.appendChild(el("span", "stars", stars(product.rating)));
    rating.appendChild(document.createTextNode(
      ` ${product.rating} (${product.review_count.toLocaleString()})`
    ));
    meta.appendChild(rating);

    if (product.available_sizes?.length) {
      const sizes = el("span");
      sizes.appendChild(el("b", null, "Sizes: "));
      sizes.appendChild(document.createTextNode(product.available_sizes.join(", ")));
      meta.appendChild(sizes);
    }
    if (product.available_colors?.length) {
      const colors = el("span");
      colors.appendChild(el("b", null, "Colours: "));
      colors.appendChild(document.createTextNode(product.available_colors.join(", ")));
      meta.appendChild(colors);
    }
    card.appendChild(meta);

    const stock = product.total_stock;
    const badge = !product.in_stock
      ? el("span", "badge badge--out", "Out of stock")
      : stock <= 12
        ? el("span", "badge badge--low", `Only ${stock} left`)
        : el("span", "badge badge--stock", "In stock");
    card.appendChild(badge);

    return card;
  }

  function orderCard(order) {
    const card = el("article", "order");

    const head = el("div", "order__head");
    const number = el("div", "order__number");
    number.appendChild(document.createTextNode(`Order ${order.order_number} `));
    number.appendChild(el("span", null, `· ${formatDate(order.placed_at)}`));
    head.appendChild(number);

    const pill = el("span", "status-pill", order.status_label || order.status);
    pill.dataset.status = order.status;
    head.appendChild(pill);
    card.appendChild(head);

    const body = el("div", "order__body");
    if (order.delivery_message) {
      body.appendChild(el("p", "order__delivery", order.delivery_message));
    }

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
          `${formatDate(event.occurred_at)}${event.location ? " · " + event.location : ""}`
        ));
        list.appendChild(item);
      }
      body.appendChild(list);
    }

    card.appendChild(body);
    return card;
  }

  /** The purchase confirmation card.
   *  The confirmation token travels server -> browser -> server and is never
   *  reproduced by the model, so a purchase requires this click. */
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

    for (const warning of quote.warnings || []) {
      body.appendChild(el("p", "quote__warning", warning));
    }
    card.appendChild(body);

    const actions = el("div", "quote__actions");
    const confirm = el("button", "button button--primary", `Confirm and pay ${quote.total.display}`);
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
          `${receipt.total.display} paid by ${receipt.payment_method.replace(/_/g, " ")}. ` +
          `Estimated delivery ${receipt.estimated_delivery_readable}.`));
        card.replaceWith(done);
        announce(`Order ${receipt.order_number} confirmed.`);
        refreshCart();
      } catch (error) {
        confirm.disabled = cancel.disabled = false;
        confirm.textContent = `Confirm and pay ${quote.total.display}`;
        const failure = el("p", "quote__warning", error.message);
        body.appendChild(failure);
        announce(`Could not place the order. ${error.message}`);
      }
    });

    cancel.addEventListener("click", () => {
      const note = el("p", "panel__empty", "Purchase cancelled. Your bag is unchanged.");
      card.replaceWith(note);
    });

    actions.appendChild(confirm);
    actions.appendChild(cancel);
    card.appendChild(actions);
    return card;
  }

  /* --------------------------------------------------------------- trace */

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
          label.appendChild(el("span", "badge", "writes"));
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
    const toolCalls = payload.trace.filter((s) => s.kind === "tool_call").length;
    summary.appendChild(el("span", null, plural(toolCalls, "backend call")));
    summary.appendChild(el("span", null, `${payload.latency_ms}ms end to end`));
    summary.appendChild(el("span", null, payload.grounded ? "Grounded" : "Ungrounded, answer withheld"));
    summary.appendChild(el("span", null, payload.model));
    return summary;
  }

  /* ------------------------------------------------------------ messages */

  function announce(text) {
    dom.status.textContent = text;
    window.setTimeout(() => {
      if (dom.status.textContent === text) dom.status.textContent = "";
    }, 6000);
  }

  function addUserMessage(text) {
    const node = document.getElementById("tpl-message-user").content.cloneNode(true);
    node.querySelector(".message__bubble").textContent = text;
    dom.transcript.appendChild(node);
    scrollToEnd();
  }

  function addTyping() {
    const node = document.getElementById("tpl-typing").content.cloneNode(true);
    const article = node.querySelector(".message");
    dom.transcript.appendChild(node);
    scrollToEnd();

    // Rotate the status line so a multi-second turn still reads as progress.
    const notes = ["Checking our systems", "Looking that up", "Reading the results"];
    let index = 0;
    const noteNode = article.querySelector(".typing__note");
    const timer = window.setInterval(() => {
      index = (index + 1) % notes.length;
      if (noteNode) noteNode.textContent = notes[index];
    }, 2200);

    return () => { window.clearInterval(timer); article.remove(); };
  }

  function addAssistantMessage(payload) {
    const node = document.getElementById("tpl-message-assistant").content.cloneNode(true);
    const article = node.querySelector(".message");
    const bubble = node.querySelector(".message__bubble");
    const cards = node.querySelector(".message__cards");
    const citations = node.querySelector(".message__citations");
    const toggle = node.querySelector(".message__trace-toggle");
    const tracePane = node.querySelector(".message__trace");

    bubble.appendChild(renderMarkdown(payload.reply));
    if (payload.blocked) article.classList.add("message--blocked");
    if (!payload.grounded) article.classList.add("message--ungrounded");

    if (payload.products?.length) {
      const grid = el("div", "card-grid");
      payload.products.forEach((p) => grid.appendChild(productCard(p)));
      cards.appendChild(grid);
    }
    payload.orders?.forEach((order) => cards.appendChild(orderCard(order)));
    if (payload.checkout_quote) cards.appendChild(quoteCard(payload.checkout_quote));

    if (payload.citations?.length) {
      citations.hidden = false;
      payload.citations.forEach((c) => citations.appendChild(el("span", "citation", c)));
    }

    tracePane.appendChild(traceList(payload.trace));
    tracePane.appendChild(traceSummary(payload));

    const traceLabel = toggle.querySelector(".trace-label");
    const toolCalls = payload.trace.filter((s) => s.kind === "tool_call").length;
    const openLabel = toolCalls
      ? `Show the ${plural(toolCalls, "backend call")} behind this`
      : "Show how this was answered";
    traceLabel.textContent = openLabel;

    toggle.addEventListener("click", () => {
      const open = toggle.getAttribute("aria-expanded") === "true";
      toggle.setAttribute("aria-expanded", String(!open));
      tracePane.hidden = open;
      traceLabel.textContent = open ? openLabel : "Hide the trace";
    });

    wireFeedback(node.querySelector(".message__feedback"), payload.turn_id);

    dom.transcript.appendChild(node);
    scrollToEnd();
  }

  function wireFeedback(group, turnId) {
    const buttons = Array.from(group.querySelectorAll(".feedback-button"));
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
            body: JSON.stringify({
              session_id: state.sessionId || "", turn_id: turnId, rating, reason, comment: "",
            }),
          });
          announce("Thanks, that is recorded against this exact turn.");
        } catch {
          announce("Could not record that feedback.");
        }
      });
    });
  }

  function scrollToEnd() {
    window.requestAnimationFrame(() => {
      dom.transcript.scrollTop = dom.transcript.scrollHeight;
    });
  }

  /* ---------------------------------------------------------------- send */

  async function send(text) {
    const message = text.trim();
    if (!message || state.busy) return;

    if (dom.welcome) { dom.welcome.remove(); dom.welcome = null; }

    state.busy = true;
    dom.sendButton.disabled = true;
    dom.input.value = "";
    resizeInput();
    dom.status.textContent = "";

    addUserMessage(message);
    const removeTyping = addTyping();

    try {
      const payload = await request(API.chat, {
        method: "POST",
        body: JSON.stringify({ message }),
      });
      removeTyping();
      addAssistantMessage(payload);
      state.lastTrace = payload;
      renderTracePanel(payload);
      if (payload.cart) renderCart(payload.cart); else refreshCart();
    } catch (error) {
      removeTyping();
      addAssistantMessage({
        reply: `I could not reach the assistant service. ${error.message}`,
        trace: [], products: [], orders: [], citations: [],
        turn_id: "error", latency_ms: 0, model: "-", grounded: true, blocked: true,
      });
    } finally {
      state.busy = false;
      dom.sendButton.disabled = false;
      dom.input.focus();
    }
  }

  /* -------------------------------------------------------------- panels */

  function renderCart(cart) {
    clear(dom.cartBody);

    if (!cart || !cart.lines.length) {
      dom.cartBody.appendChild(el("p", "panel__empty",
        "Your bag is empty. Ask me to add something and it will appear here."));
      dom.cartCount.hidden = true;
      return;
    }

    dom.cartCount.hidden = false;
    dom.cartCount.textContent = cart.item_count;

    for (const line of cart.lines) {
      const row = el("div", "cart-line");
      const left = el("div");
      left.appendChild(el("div", "cart-line__name", line.product_name));
      left.appendChild(el("div", "cart-line__meta",
        `${line.quantity} x ${line.size} · ${line.color} · ${line.unit_price.display}`));
      const remove = el("button", "cart-line__remove", "Remove");
      remove.type = "button";
      remove.setAttribute("aria-label", `Remove ${line.product_name} from your bag`);
      remove.addEventListener("click", async () => {
        remove.disabled = true;
        try {
          renderCart(await request(API.removeLine(line.variant_id), { method: "DELETE" }));
          announce(`${line.product_name} removed from your bag.`);
        } catch (error) {
          remove.disabled = false;
          announce(error.message);
        }
      });
      left.appendChild(remove);
      row.appendChild(left);
      row.appendChild(el("div", "cart-line__price", line.line_total.display));
      dom.cartBody.appendChild(row);
    }

    const totals = el("div", "cart-totals");
    for (const [label, value] of [
      ["Subtotal", cart.subtotal.display],
      ["Shipping", cart.shipping.amount_cents === 0 ? "Free" : cart.shipping.display],
      ["Tax", cart.tax.display],
    ]) {
      const row = el("div", "cart-totals__row");
      row.appendChild(el("span", null, label));
      row.appendChild(el("span", null, value));
      totals.appendChild(row);
    }
    const total = el("div", "cart-totals__row cart-totals__row--total");
    total.appendChild(el("span", null, "Total"));
    total.appendChild(el("span", null, cart.total.display));
    totals.appendChild(total);
    dom.cartBody.appendChild(totals);

    if (cart.note) dom.cartBody.appendChild(el("p", "cart-note", cart.note));
  }

  async function refreshCart() {
    try { renderCart(await request(API.cart)); } catch { /* panel keeps its last good state */ }
  }

  function renderTracePanel(payload) {
    clear(dom.traceBody);
    dom.traceBody.appendChild(traceList(payload.trace));
    dom.traceBody.appendChild(traceSummary(payload));
  }

  async function renderGovernance() {
    try {
      const metrics = await request(API.metrics);
      clear(dom.governanceBody);

      const grid = el("div", "metric-grid");
      const tiles = [
        [metrics.tool_calls_total.toLocaleString(), "Backend calls"],
        [metrics.guardrail_events_total.toLocaleString(), "Guardrail checks"],
        [metrics.guardrail_blocks.toLocaleString(), "Blocked"],
        [`${(metrics.guardrail_block_rate * 100).toFixed(1)}%`, "Block rate"],
      ];
      for (const [value, label] of tiles) {
        const tile = el("div", "metric");
        tile.appendChild(el("div", "metric__value", value));
        tile.appendChild(el("div", "metric__label", label));
        grid.appendChild(tile);
      }
      dom.governanceBody.appendChild(grid);

      if (metrics.tools.length) {
        dom.governanceBody.appendChild(el("p", "section-title", "Tool usage"));
        const busiest = Math.max(...metrics.tools.map((t) => t.calls), 1);
        for (const tool of metrics.tools.slice(0, 9)) {
          const row = el("div", "bar-row");
          row.appendChild(el("span", "bar-row__name", tool.tool));
          const track = el("div", "bar-row__track");
          const fill = el("div", "bar-row__fill");
          fill.style.width = `${(tool.calls / busiest) * 100}%`;
          track.appendChild(fill);
          row.appendChild(track);
          row.appendChild(el("span", "bar-row__value", tool.calls));
          dom.governanceBody.appendChild(row);
        }
      }

      if (metrics.guardrails.length) {
        dom.governanceBody.appendChild(el("p", "section-title", "Guardrail decisions"));
        for (const rule of metrics.guardrails.slice(0, 10)) {
          const row = el("div", "rule-row");
          row.appendChild(el("span", "rule-row__name", `${rule.stage}/${rule.rule}`));
          const action = el("span", "rule-row__action", `${rule.action} · ${rule.count}`);
          action.dataset.action = rule.action;
          row.appendChild(action);
          dom.governanceBody.appendChild(row);
        }
      }
    } catch (error) {
      clear(dom.governanceBody);
      dom.governanceBody.appendChild(el("p", "panel__empty", `Could not load metrics. ${error.message}`));
    }
  }

  /* ----------------------------------------------------------------- tabs */

  function selectTab(tab) {
    for (const other of dom.tabs) {
      const selected = other === tab;
      other.setAttribute("aria-selected", String(selected));
      other.tabIndex = selected ? 0 : -1;
      document.getElementById(other.getAttribute("aria-controls")).hidden = !selected;
    }
    if (tab.id === "tab-governance") renderGovernance();
  }

  dom.tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => selectTab(tab));
    // Roving tabindex: arrows move between tabs, Home and End jump to the ends.
    tab.addEventListener("keydown", (event) => {
      const keys = { ArrowRight: 1, ArrowLeft: -1 };
      let next = null;
      if (event.key in keys) next = dom.tabs[(index + keys[event.key] + dom.tabs.length) % dom.tabs.length];
      else if (event.key === "Home") next = dom.tabs[0];
      else if (event.key === "End") next = dom.tabs[dom.tabs.length - 1];
      if (next) { event.preventDefault(); next.focus(); selectTab(next); }
    });
  });

  /* --------------------------------------------------------------- theme */

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    dom.themeToggle.setAttribute(
      "aria-label", theme === "dark" ? "Switch to light theme" : "Switch to dark theme"
    );
    try { localStorage.setItem("aurelia-theme", theme); } catch { /* private mode */ }
  }

  dom.themeToggle.addEventListener("click", () => {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  });

  /* ------------------------------------------------------------ composer */

  function resizeInput() {
    dom.input.style.height = "auto";
    dom.input.style.height = `${Math.min(dom.input.scrollHeight, 160)}px`;
  }

  dom.input.addEventListener("input", resizeInput);
  dom.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send(dom.input.value);
    }
  });

  dom.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    send(dom.input.value);
  });

  document.addEventListener("click", (event) => {
    const suggestion = event.target.closest(".suggestion");
    if (suggestion) send(suggestion.dataset.prompt);
  });

  dom.refreshGov.addEventListener("click", renderGovernance);

  /* ----------------------------------------------------------------- init */

  (async function init() {
    try {
      const stored = localStorage.getItem("aurelia-theme");
      applyTheme(stored || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
    } catch {
      applyTheme("light");
    }

    try {
      const session = await request(API.session);
      state.sessionId = session.session_id;
      dom.identityName.textContent = session.customer_name;
      dom.modelName.textContent = session.model;
      document.getElementById("identity-chip").title =
        `Acting for ${session.customer_name} (${session.customer_public_id}, ${session.loyalty_tier} tier). ` +
        `Only this customer's orders are visible.`;
    } catch {
      dom.identityName.textContent = "Guest";
      dom.modelName.textContent = "offline";
    }

    refreshCart();
    resizeInput();
    dom.input.focus();

    // Deep link: /#ask=<url-encoded question> opens the page with that question
    // already asked. Useful for sharing a worked example, and it is how the
    // screenshots in the README are captured reproducibly.
    const hash = window.location.hash;
    if (hash.startsWith("#ask=")) {
      const question = decodeURIComponent(hash.slice(5)).trim();
      if (question) send(question);
    }
    // Support ?theme=light|dark for the same reason.
    const themeParam = new URLSearchParams(window.location.search).get("theme");
    if (themeParam === "light" || themeParam === "dark") applyTheme(themeParam);
  })();
})();
