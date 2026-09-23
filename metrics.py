from collections import Counter
import re
class NERMetric:
    _ENTITY_RE = re.compile(
        r"^(.+?):\s*GENE\s*$",
        re.IGNORECASE,
    )

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def parse_entities(self, text):
        text = str(text).strip()
        if not text or text == "无实体":
            return Counter()

        # 清理代码块
        cleaned = re.sub(
            r"^```(?:json|text)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        entities = Counter()

        for line in cleaned.splitlines():
            line = line.strip()

            # 清理 Markdown 列表和编号
            line = re.sub(
                r"^\s*(?:[-*]|\d+[.)])\s*",
                "",
                line,
            )

            if not line or line == "无实体":
                continue

            match = self._ENTITY_RE.match(line)
            if not match:
                continue

            entity_name = " ".join(match.group(1).split())
            if entity_name:
                entities[(entity_name, "GENE")] += 1

        return entities

    def update(self, pred_text, gold_text):
        pred_entities = self.parse_entities(pred_text)
        gold_entities = self.parse_entities(gold_text)

        tp = sum((pred_entities & gold_entities).values())

        self.tp += tp
        self.fp += sum(pred_entities.values()) - tp
        self.fn += sum(gold_entities.values()) - tp

    def compute(self):
        if self.tp + self.fp == 0:
            precision = 0
        else:
            precision = self.tp / (self.tp + self.fp)

        if self.tp + self.fn == 0:
            recall = 0
        else:
            recall = self.tp / (self.tp + self.fn)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
