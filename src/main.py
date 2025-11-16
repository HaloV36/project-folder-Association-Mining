# src/main.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Set, List
import time

from data_io import load_transactions_csv, load_products_csv, df_to_transactions, basic_stats
from preprocessing.preprocess import preprocess_transactions, standardize_item_name
from algorithms.apriori import apriori, generate_association_rules
from algorithms.eclat import eclat, generate_association_rules_eclat

# ---------- Helper ----------
def transactions_to_df(transactions: Dict[str, Set[str]]) -> pd.DataFrame:
    rows = []
    for tid, items in transactions.items():
        rows.append({"tid": tid, "items": ", ".join(sorted(items))})
    return pd.DataFrame(rows)

def rules_for_item(rules: List[dict], item: str) -> List[dict]:
    item_std = standardize_item_name(item)
    filtered = []
    for r in rules:
        if item_std in r["antecedent"]:
            filtered.append(r)
    # sort by confidence, then support
    filtered.sort(key=lambda x: (x["confidence"], x["support"]), reverse=True)
    return filtered

# ---------- Streamlit App ----------
st.set_page_config(page_title="Supermarket Association Mining", layout="wide")

st.title("Interactive Supermarket Simulation with Association Rule Mining")

st.markdown("""
This app lets you:
1. Create shopping transactions manually or import from CSV  
2. Clean the data (preprocessing)  
3. Run **Apriori** and **Eclat**  
4. Explore association rules and get **product recommendations**
""")

# Sidebar: parameters
st.sidebar.header("Algorithm Parameters")
min_support = st.sidebar.slider("Minimum Support", 0.01, 1.0, 0.2, 0.01)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.01, 1.0, 0.5, 0.01)

# ---- Load product list ----
st.sidebar.header("Data Files")
products_file = st.sidebar.text_input("products.csv path", "data/products.csv")
transactions_file_default = "data/sample_transactions.csv"

valid_products: Set[str] = set()
product_df = None
try:
    product_df = load_products_csv(products_file)

    # Standardized product names
    name_tokens = {standardize_item_name(str(n)) for n in product_df["name"].astype(str)}

    # Standardized product IDs
    id_tokens = {standardize_item_name(str(pid)) for pid in product_df["product_id"].astype(str)}

    # Accept either IDs or names as valid
    valid_products = name_tokens | id_tokens

    st.sidebar.success(
        f"Loaded {len(product_df)} products "
        f"(unique valid tokens: {len(valid_products)})"
    )
except Exception as e:
    st.sidebar.warning(f"Could not load products file: {e}")


# ---- Session State for Manual Transactions ----
if "manual_transactions" not in st.session_state:
    st.session_state.manual_transactions = {}  # tid -> set(items)
if "next_tid" not in st.session_state:
    st.session_state.next_tid = 1

# Predefined 10+ products for UI buttons (names should exist in products.csv ideally)
DEFAULT_PRODUCTS = [
    "Milk", "Bread", "Eggs", "Butter", "Cheese",
    "Apple", "Banana", "Chicken", "Rice", "Pasta",
    "Tomato", "Yogurt", "Cereal", "Coffee", "Tea", "Steak",
]

# ---- Tabs ----
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Shopping System",
    "2. Preprocessing",
    "3. Association Mining",
    "4. Recommendations"
])

# ---------- TAB 1: Shopping System ----------
with tab1:
    st.header("Manual Transaction Creation")

    st.write("Click items to add them to the current transaction:")

    cols = st.columns(4)
    current_items = st.session_state.get("current_items", set())

    for idx, prod in enumerate(DEFAULT_PRODUCTS):
        col = cols[idx % 4]
        if col.button(prod):
            current_items.add(prod)
    st.session_state.current_items = current_items

    st.write("Current transaction items:", ", ".join(sorted(current_items)) or "(none)")

    col_a, col_b = st.columns(2)
    if col_a.button("Save Transaction"):
        if current_items:
            tid = f"m{st.session_state.next_tid}"
            st.session_state.next_tid += 1
            st.session_state.manual_transactions[tid] = set(current_items)
            st.session_state.current_items = set()
            st.success(f"Saved transaction {tid}")
        else:
            st.warning("Cannot save empty transaction.")
    if col_b.button("Clear Current Items"):
        st.session_state.current_items = set()

    st.subheader("Manual Transactions")

    manual_tx = st.session_state.manual_transactions

    if manual_tx:
        st.dataframe(transactions_to_df(manual_tx))

        # --- Delete a single manual transaction by ID ---
        st.markdown("**Delete a single transaction**")
        col_del1, col_del2 = st.columns([2, 1])

        with col_del1:
            tid_to_delete = st.text_input(
                "Enter transaction ID to delete (e.g., m1, m2, ...), Make sure to click 'Delete Transaction' TWICE to confirm.",
                key="tid_to_delete_input"
            )

        with col_del2:
            if st.button("Delete Transaction"):
                if tid_to_delete in manual_tx:
                    del manual_tx[tid_to_delete]
                    st.success(f"Deleted manual transaction {tid_to_delete}")
                else:
                    st.warning(f"Transaction ID '{tid_to_delete}' not found among manual transactions.")

        st.markdown("---")

        # --- Clear ALL manual transactions ---
        if st.button("Clear ALL Manual Transactions"):
            st.session_state.manual_transactions = {}
            st.success("All manual transactions have been cleared.")

    else:
        st.info("No manual transactions yet.")

    st.markdown("---")
    st.header("CSV Data Import")

    uploaded = st.file_uploader("Upload sample_transactions.csv", type=["csv"])
    if uploaded is not None:
        try:
            df_csv = pd.read_csv(uploaded)
            st.session_state.import_df = df_csv
            # Convert to long format & dict
            df_norm = load_transactions_csv(uploaded)
            imported_trans = df_to_transactions(df_norm)
            st.session_state.import_transactions = imported_trans
            st.success(f"Imported {len(imported_trans)} transactions from CSV.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        # optionally auto-load default file if exists
        try:
            df_norm = load_transactions_csv(transactions_file_default)
            imported_trans = df_to_transactions(df_norm)
            st.session_state.import_transactions = imported_trans
            st.info(f"Using default file: {transactions_file_default} ({len(imported_trans)} transactions).")
        except Exception:
            st.warning("No CSV loaded. Only manual transactions will be used.")

    # Combine manual + imported
    existing_imported = st.session_state.get("import_transactions", {})
    all_transactions_raw = {**existing_imported, **st.session_state.manual_transactions}
    st.subheader("Current Transactions (RAW)")
    if all_transactions_raw:
        st.write(basic_stats(all_transactions_raw))
        st.dataframe(transactions_to_df(all_transactions_raw))
    else:
        st.info("No transactions available yet.")

# ---------- TAB 2: Preprocessing ----------
with tab2:
    st.header("Data Preprocessing")
    all_transactions_raw = st.session_state.get("import_transactions", {})
    all_transactions_raw = {**all_transactions_raw, **st.session_state.manual_transactions}

    if not all_transactions_raw:
        st.warning("No transactions to preprocess. Add or import data in Tab 1.")
    else:
        if st.button("Run Preprocessing"):
            cleaned, report = preprocess_transactions(
                all_transactions_raw,
                valid_products or {standardize_item_name(p) for p in DEFAULT_PRODUCTS}
            )

            st.session_state.cleaned_transactions = cleaned
            st.session_state.preprocessing_report = report
            st.success("Preprocessing completed.")

        if "preprocessing_report" in st.session_state:
            st.subheader("Preprocessing Report")
            st.text(st.session_state.preprocessing_report)

            st.subheader("Cleaned Transactions")
            cleaned = st.session_state.cleaned_transactions
            st.write(basic_stats(cleaned))
            st.dataframe(transactions_to_df(cleaned))
        else:
            st.info("Click 'Run Preprocessing' to clean the data.")

# ---------- TAB 3: Association Mining ----------
with tab3:
    st.header("Association Rule Mining: Apriori vs Eclat")
    cleaned = st.session_state.get("cleaned_transactions", None)
    if not cleaned:
        st.warning("Please run preprocessing first (Tab 2).")
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Run Apriori"):
                frequents_a, time_a = apriori(cleaned, min_support)
                rules_a = generate_association_rules(frequents_a, min_confidence)
                st.session_state.apriori_results = {
                    "frequent_itemsets": frequents_a,
                    "rules": rules_a,
                    "time": time_a
                }
        with col2:
            if st.button("Run Eclat"):
                frequents_e, time_e = eclat(cleaned, min_support)
                rules_e = generate_association_rules_eclat(frequents_e, min_confidence)
                st.session_state.eclat_results = {
                    "frequent_itemsets": frequents_e,
                    "rules": rules_e,
                    "time": time_e
                }

        # Show comparison
        st.subheader("Performance Comparison")
        apr = st.session_state.get("apriori_results")
        ecl = st.session_state.get("eclat_results")

        data_rows = []
        if apr:
            data_rows.append({
                "Algorithm": "Apriori",
                "Execution Time (ms)": round(apr["time"] * 1000, 2),
                "Frequent Itemsets": len(apr["frequent_itemsets"]),
                "Rules Generated": len(apr["rules"]),
            })
        if ecl:
            data_rows.append({
                "Algorithm": "Eclat",
                "Execution Time (ms)": round(ecl["time"] * 1000, 2),
                "Frequent Itemsets": len(ecl["frequent_itemsets"]),
                "Rules Generated": len(ecl["rules"]),
            })

        if data_rows:
            df_perf = pd.DataFrame(data_rows)
            st.dataframe(df_perf)
        else:
            st.info("Run at least one algorithm to see the comparison.")

        # Optionally show raw rules (technical view)
        with st.expander("Show Detailed Rules (Technical View)"):
            algo_choice = st.selectbox("Algorithm", ["Apriori", "Eclat"])
            res = st.session_state.get("apriori_results" if algo_choice == "Apriori" else "eclat_results")
            if res:
                rules = res["rules"]
                for r in rules[:100]:  # limit to 100
                    st.write(
                        f"{set(r['antecedent'])} → {set(r['consequent'])} | "
                        f"support={r['support']:.2f}, confidence={r['confidence']:.2f}, lift={r['lift']:.2f}"
                    )
            else:
                st.info("Run the selected algorithm first.")

# ---------- TAB 4: User-Friendly Recommendations ----------
with tab4:
    st.header("Product Recommendations (User-Friendly View)")
    cleaned = st.session_state.get("cleaned_transactions", None)

    if not cleaned:
        st.warning("Need cleaned data and mined rules first (Tab 2 + Tab 3).")
    else:
        # Get results (if any) from Tab 3
        apr = st.session_state.get("apriori_results")
        ecl = st.session_state.get("eclat_results")

        # Build dropdown options based on what actually exists
        options = []
        if apr is not None:
            options.append("Apriori")
        if ecl is not None:
            options.append("Eclat")

        if not options:
            st.info("Run Apriori and/or Eclat in Tab 3 before using recommendations.")
        else:
            # Dropdown to choose algorithm (default is Apriori if present)
            algo_choice = st.selectbox(
                "Choose algorithm for recommendations:",
                options,
                index=0  # default to first option (Apriori if it exists)
            )

            # Select rules based on dropdown
            if algo_choice == "Apriori":
                rules_source = (apr or {}).get("rules", [])
            else:
                rules_source = (ecl or {}).get("rules", [])

            if not rules_source:
                st.info(f"No rules generated by {algo_choice}. Try lowering min support/confidence and rerun it in Tab 3.")
            else:
                st.caption(f"Using rules from **{algo_choice}**")

                # Build item dropdown from cleaned data
                all_items = sorted({i for t in cleaned.values() for i in t})
                selected_item = st.selectbox("Select a product:", all_items)

                if selected_item:
                    rel_rules = rules_for_item(rules_source, selected_item)
                    if not rel_rules:
                        st.info("No strong associations found for this item with the current thresholds.")
                    else:
                        st.subheader(f"Customers who bought **{selected_item}** also bought:")
                        for r in rel_rules[:10]:
                            conf_pct = r["confidence"] * 100
                            sup_pct = r["support"] * 100
                            consequent_str = ", ".join(sorted(r["consequent"]))
                            strength_label = (
                                "Strong" if conf_pct >= 70
                                else "Moderate" if conf_pct >= 40
                                else "Weak"
                            )
                            bar = "█" * int(conf_pct // 5)
                            st.markdown(
                                f"- **{consequent_str}**: {conf_pct:.1f}% of the time "
                                f"(support {sup_pct:.1f}%)  {bar}  *({strength_label})*"
                            )

                        st.markdown("---")
                        st.markdown(
                            f"**Business Recommendation:** Based on **{algo_choice}** rules, "
                            f"consider placing **{selected_item}** near its top associated item(s), "
                            f"or creating bundles like **{selected_item} + [top associated item]**."
                        )
