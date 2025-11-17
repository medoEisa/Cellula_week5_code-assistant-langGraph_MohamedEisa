import re

class IntentClassifier:
    def __init__(self):
        self.mapping = {
            'generate': ['generate','create','implement','write','make','function','class','def','return','script','build'],
            'explain': ['explain','why','how','what','describe','walkthrough','difference','meaning'],
            'chat': ['^hi$','^hello$','^hey$','help','thanks','thank you','how are you','what can you do','tell me about yourself','who are you']
        }

    def infer(self, text: str) -> str:
        t = text.lower()
        scores = {k:0 for k in self.mapping}
        for intent, keys in self.mapping.items():
            for kw in keys:
                try:
                    pattern = r'\b' + kw + r'\b' if not (kw.startswith('^') or kw.endswith('$')) else kw
                    if re.search(pattern, t):
                        scores[intent] += 1
                except re.error:
                    if kw in t:
                        scores[intent] += 1
        if re.search(r'\b(def |class |import |lambda |->|\:\n)', text):
            scores['generate'] += 1

        best = max(scores.items(), key=lambda kv: kv[1])
        return best[0] if best[1] > 0 else 'explain'
