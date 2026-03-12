import itertools
import pandas as pd


def get_frequent_itemsets(transactions, min_support=0.15):
    all_items = sorted(set(item for trx in transactions for item in trx))
    oht = pd.DataFrame([{item: (item in trx) for item in all_items} for trx in transactions])

    num_trx = len(transactions)
    support_map = {}

    item_counts = {}
    for trx in transactions:
        unique_items = set(trx)
        for item in unique_items:   
            item_counts[item] = item_counts.get(item, 0) + 1

    frequents = [
        frozenset([item]) for item, cnt in item_counts.items() if cnt / num_trx >= min_support
    ]
    for itemset in frequents:
        support_map[itemset] = item_counts[next(iter(itemset))] / num_trx

    k = 2
    all_frequents = list(frequents)

    def generate_candidates(prev_frequents, k_size):
        prev_list = list(prev_frequents)
        candidates = set()
        for i in range(len(prev_list)):
            for j in range(i + 1, len(prev_list)):
                union = prev_list[i] | prev_list[j]
                if len(union) == k_size:
                    if all((union - frozenset([x])) in prev_frequents for x in union):
                        candidates.add(union)
        return candidates

    current_frequents = set(frequents)
    while current_frequents:
        candidates = generate_candidates(current_frequents, k)
        counts = {c: 0 for c in candidates}
        for trx in transactions:
            trx_set = set(trx)
            for cand in candidates:
                if cand.issubset(trx_set):
                    counts[cand] += 1

        current_frequents = set()
        for cand, cnt in counts.items():
            support = cnt / num_trx
            if support >= min_support:
                current_frequents.add(cand)
                support_map[cand] = support
        all_frequents.extend(current_frequents)
        k += 1

    frequent_df = pd.DataFrame(
        [{"support": support_map[fs], "itemsets": fs} for fs in all_frequents]
    ).sort_values("support", ascending=False).reset_index(drop=True)

    return frequent_df, oht


def get_association_rules(frequent_itemsets, min_confidence=0.2, min_lift=1.0):
    """Compute association rules from frequent itemsets."""
    support_lookup = {frozenset(row.itemsets): row.support for row in frequent_itemsets.itertuples()}
    rules = []

    for itemset, supp in support_lookup.items():
        if len(itemset) < 2:
            continue
        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                conf = supp / support_lookup[antecedent]
                lift = conf / support_lookup[consequent]
                if conf >= min_confidence and lift >= min_lift:
                    rules.append(
                        {
                            "antecedents": antecedent,
                            "consequents": consequent,
                            "support": supp,
                            "confidence": conf,
                            "lift": lift,
                        }
                    )

    rules_df = pd.DataFrame(rules)
    if not rules_df.empty:
        rules_df = rules_df[
            (rules_df["antecedents"].apply(lambda x: len(x) == 1)) &
            (rules_df["consequents"].apply(lambda x: len(x) == 1))
        ]
        
        rules_df = rules_df.sort_values("lift", ascending=False).reset_index(drop=True)
    return rules_df