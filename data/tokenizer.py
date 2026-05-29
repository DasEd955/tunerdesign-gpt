from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens = list(corpus)
        merges = list()
        for _ in range(num_merges):

            # Edge Case; only 1 token left, no adjacent pair exists
            if len(tokens) < 2:
                break
            
            # Count adjacent pair frequencies
            pairs = dict()
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1
            # Edge Case; if no pairs exist, stop
            if not pairs:
                break

            # Find the most frequent pair (tiebreak: lexicographically smallest) 
                # Note: Greedy Algorithm decision-making
            best_count = max(pairs.values())
            candidates = sorted(pair for pair, count in pairs.items() if count == best_count)
            best = candidates[0]

            merges.append([best[0], best[1]])

            # Merge all non-overlapping occurences from left to right
            new_tokens = list()
            i = 0
            # Use manual pointer movement since merges may skip tokens
            while i < len(tokens):
                # If current adjacent pair matches the best pair:
                    # Merge them into a single token & skip both
                if i < len(tokens) - 1 and tokens[i] == best[0] and tokens[i + 1] == best[1]:
                    new_tokens.append(best[0] + best[1])
                    i += 2
                # Otherwise keep current token & move normally
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return merges 
            
