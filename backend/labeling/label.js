// filepath: /Users/jameshan/Documents/github/giftfindr/faiss/label.js
let gold,
  products,
  idx = 0;
let queueIndices = [],
  cIdx = 0;
const qEl = document.getElementById("query"),
  productsEl = document.getElementById("products"),
  filterEl = document.getElementById("filter"),
  filterByQueryEl = document.getElementById("filterByQuery"),
  minPriceEl = document.getElementById("minPrice"),
  maxPriceEl = document.getElementById("maxPrice"),
  prevBtn = document.getElementById("prev"),
  nextBtn = document.getElementById("next"),
  overwriteBtn = document.getElementById("overwrite"),
  removeBtn = document.getElementById("remove");

// Initialize gold as empty array if fetch fails
gold = [];
products = [];

Promise.all([
  fetch("../data/gold.json")
    .then((r) => r.json())
    .catch(() => []),
  fetch("../data/products.json").then((r) => r.json()),
]).then(([g, p]) => {
  gold = g || [];
  products = p;
  queueIndices = gold.map((_, i) => i);
  if (!queueIndices.length) {
    qEl.textContent =
      "No queries found! Use FAISS Search tab to add new queries.";
    productsEl.innerHTML = "";
    return;
  }
  cIdx = 0;
  idx = queueIndices[cIdx];
  render();
});

function render() {
  if (!gold || gold.length === 0) {
    qEl.textContent =
      "No queries found! Use FAISS Search tab to add new queries.";
    productsEl.innerHTML = "";
    return;
  }

  const item = gold[idx];
  qEl.textContent = `${cIdx + 1}/${queueIndices.length}: ${item.query}`;
  // set price filter defaults from any "$NN" in the query
  const m = item.query.match(/\$(\d+)/);
  minPriceEl.value = 0;
  maxPriceEl.value = m ? m[1] : "";
  renderProducts(item);
}

function renderProducts(item) {
  productsEl.innerHTML = "";
  const filter = filterEl.value.toLowerCase();
  const min = parseFloat(minPriceEl.value) || 0;
  const max = parseFloat(maxPriceEl.value) || Infinity;

  // highlight setup for query words
  const stopWords = new Set([
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "for",
    "of",
    "in",
    "on",
    "to",
    "with",
    "my",
    "who",
    "gift",
  ]);
  const words = item.query
    .split(/\s+/)
    .map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .filter((w) => !stopWords.has(w.toLowerCase()));
  const regex = new RegExp(`\\b(${words.join("|")})\\b`, "gi");

  // highlight setup for filter words
  let filterRegex = null;
  if (filter && filter.trim()) {
    const filterWords = filter
      .trim()
      .split(/\s+/)
      .map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
      .filter((w) => w.length > 1); // Only words longer than 1 char
    if (filterWords.length > 0) {
      filterRegex = new RegExp(`\\b(${filterWords.join("|")})\\b`, "gi");
    }
  }

  products
    .filter((p) => p.title.toLowerCase().includes(filter))
    .filter((p) => p.price >= min && p.price <= max)
    .filter((p) => {
      if (!filterByQueryEl.checked) return true;
      const text = `${p.title} ${p.category || ""}`.toLowerCase();
      return words.some((w) => new RegExp(`\\b${w}\\b`, "i").test(text));
    })
    .forEach((p) => {
      const div = document.createElement("div");
      div.className = "product";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = p.id;
      if (item.relevant_ids.includes(p.id)) cb.checked = true;
      cb.addEventListener("change", () => {
        const ids = item.relevant_ids;
        if (cb.checked) ids.push(p.id);
        else item.relevant_ids = ids.filter((id) => id !== p.id);
      });
      // wrap in label so clicking text toggles checkbox:
      const lbl = document.createElement("label");
      lbl.appendChild(cb);

      // Apply query word highlights
      let highlightedTitle = p.title.replace(
        regex,
        (m) => `<span class="highlight">${m}</span>`
      );

      // Apply filter word highlights (if any)
      if (filterRegex) {
        // Need to work with HTML, so use careful regex replacement
        highlightedTitle = highlightedTitle.replace(
          filterRegex,
          function (match) {
            // Skip if already inside a highlight span
            const prevHtml = highlightedTitle.substring(
              0,
              highlightedTitle.indexOf(match)
            );
            const openTags = (prevHtml.match(/<span/g) || []).length;
            const closeTags = (prevHtml.match(/<\/span>/g) || []).length;
            if (openTags > closeTags) return match;
            return `<span class="filter-highlight">${match}</span>`;
          }
        );
      }

      const priceSuffix = p.price != null ? ` ($${p.price})` : "";
      const prefix = p.category ? `[${p.category}] ` : "";
      lbl.insertAdjacentHTML(
        "beforeend",
        ` ${prefix}${highlightedTitle}${priceSuffix}`
      );
      div.appendChild(lbl);
      productsEl.appendChild(div);
    });
}

filterEl.addEventListener("input", () => renderProducts(gold[idx]));
minPriceEl.addEventListener("input", () => renderProducts(gold[idx]));
maxPriceEl.addEventListener("input", () => renderProducts(gold[idx]));
filterByQueryEl.addEventListener("change", () => renderProducts(gold[idx]));

prevBtn.onclick = () => {
  if (cIdx > 0) {
    cIdx--;
    idx = queueIndices[cIdx];
    render();
  }
};

nextBtn.onclick = () => {
  // advance queue
  cIdx++;
  if (cIdx >= queueIndices.length) {
    qEl.textContent = "Done! Well done!";
    productsEl.innerHTML = "";
  } else {
    idx = queueIndices[cIdx];
    render();
  }
};

removeBtn.onclick = () => {
  // remove current query from gold
  gold.splice(idx, 1);
  // rebuild queue
  queueIndices = gold.map((_, i) => i);
  // if none left, finish
  if (!queueIndices.length) {
    qEl.textContent = "Done! Well done!";
    productsEl.innerHTML = "";
    return;
  }
  // clamp cIdx & set new idx
  cIdx = Math.min(cIdx, queueIndices.length - 1);
  idx = queueIndices[cIdx];
  render();
};

// Update the export button
overwriteBtn.onclick = () => {
  // Create gold.json
  const goldBlob = new Blob([JSON.stringify(gold, null, 2)], {
    type: "application/json",
  });
  const goldUrl = URL.createObjectURL(goldBlob);
  const goldLink = document.createElement("a");
  goldLink.href = goldUrl;
  goldLink.download = "gold.json";
  goldLink.click();
  URL.revokeObjectURL(goldUrl);

  // Show user instructions
  alert(
    "gold.json has been downloaded. To complete the overwrite:\n\n" +
      "Move gold.json to /faiss/data/\n\n" +
      "This will replace the existing file."
  );
};

// arrow‑key navigation + Del shortcut
document.addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft") prevBtn.click();
  if (e.key === "ArrowRight") nextBtn.click();
  if (e.key === "Delete") removeBtn.click();
});
