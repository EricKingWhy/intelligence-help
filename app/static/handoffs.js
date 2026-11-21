mountAdminNav("/handoffs");

const status = document.querySelector("#queue-status");
const queue = document.querySelector("#queue");
const filter = document.querySelector("#queue-filter");
const refresh = document.querySelector("#refresh");
const previous = document.querySelector("#previous");
const next = document.querySelector("#next");
const pageSize = 20;
let offset = 0;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
}

async function request(path, options) {
  const response = await fetch("/api/operator/handoffs" + path, options);
  if (!response.ok) {
    const result = await response.json().catch(() => ({}));
    throw new Error(typeof result.detail === "string" ? result.detail : "服务暂不可用，请刷新重试");
  }
  return response.json();
}

async function submit(item, action, button, resolution) {
  button.disabled = true;
  status.textContent = action === "claim" ? "正在接单…" : "正在提交处理结论…";
  const options = { method: "POST" };
  if (resolution !== undefined) {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify({ resolution });
  }
  try {
    await request("/" + encodeURIComponent(item.ticket_no) + "/" + action, options);
    await load(item.ticket_no);
  } catch (error) {
    status.textContent = error.message;
    button.disabled = false;
  }
}

function renderItem(item, operator, open) {
  const card = element("details", "", "handoff");
  card.open = open;
  const summary = element("summary");
  const heading = element("span", "", "handoff-summary");
  heading.append(element("strong", item.ticket_no), element("span", item.status, "pill"));
  summary.append(heading);
  card.append(summary);

  const body = element("div", "", "handoff-body");
  body.append(element("p", "会话 #" + item.conversation_id
    + (item.assigned_to ? " · 处理人 " + item.assigned_to : ""), "handoff-meta"));
  body.append(element("h2", "客户问题"), element("p", item.description));
  if (item.context.service_failure) {
    const descriptions = {
      rate_limited: "上游限流", timeout: "查询超时",
      knowledge_unavailable: "知识库不可用", unavailable: "上游服务不可用",
    };
    body.append(element("h2", "转接原因"), element("p",
      descriptions[item.context.service_failure.kind] || "查询中断"));
  }
  const order = item.context.order;
  body.append(element("h2", "关联订单"), element("p", order
    ? [order.order_id, order.product, order.status].filter(Boolean).join(" · ")
    : "本轮没有关联订单"));
  body.append(element("h2", "已检索的参考资料"));
  if (item.context.evidence.length) {
    const citations = element("ul");
    for (const hit of item.context.evidence) {
      citations.append(element("li", [hit.section_path, hit.question, hit.answer].filter(Boolean).join("\n")));
    }
    body.append(citations);
  } else {
    body.append(element("p", "本轮没有检索参考"));
  }

  const actions = element("div", "", "handoff-foot");
  if (item.status === "待接单") {
    const button = element("button", "接单", "btn go");
    button.type = "button";
    button.addEventListener("click", () => submit(item, "claim", button));
    actions.append(button);
  } else if (item.status === "处理中" && item.assigned_to === operator) {
    const form = element("form");
    const field = element("label", "给客户的处理结论");
    const input = element("textarea");
    input.rows = 3;
    input.maxLength = 2000;
    input.required = true;
    field.append(input);
    const button = element("button", "提交并结单", "btn go");
    form.append(field, button);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const resolution = input.value.trim();
      if (resolution) submit(item, "resolve", button, resolution);
      else { status.textContent = "请填写处理结论"; input.focus(); }
    });
    actions.append(form);
  } else if (item.resolution) {
    body.append(element("h2", "给客户的处理结论"), element("p", item.resolution));
  }
  body.append(actions);
  card.append(body);
  return card;
}

async function load(openTicket) {
  refresh.disabled = filter.disabled = previous.disabled = next.disabled = true;
  status.textContent = "正在加载人工工单…";
  try {
    const params = new URLSearchParams({ status: filter.value, limit: pageSize + 1, offset });
    const { items, operator } = await request("?" + params);
    if (!items.length && offset > 0) {
      offset = Math.max(0, offset - pageSize);
      return await load(openTicket);
    }
    queue.replaceChildren(...items.slice(0, pageSize).map(item =>
      renderItem(item, operator, item.ticket_no === openTicket)));
    status.textContent = items.length
      ? "第 " + (offset / pageSize + 1) + " 页"
      : filter.value === "open" ? "没有待处理的人工工单。" : "还没有已处理的人工工单。";
    previous.hidden = offset === 0;
    next.hidden = items.length <= pageSize;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    refresh.disabled = filter.disabled = previous.disabled = next.disabled = false;
  }
}

filter.addEventListener("change", () => { offset = 0; load(); });
refresh.addEventListener("click", () => load());
previous.addEventListener("click", () => { offset -= pageSize; load(); });
next.addEventListener("click", () => { offset += pageSize; load(); });
load();
