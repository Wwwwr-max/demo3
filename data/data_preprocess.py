import json
data_root = [
    {
        "name":"train",
        "root":"train.json"
    },
    {
        "name":"dev",
        "root":"dev.json"
    },
    {
        "name":"test",
        "root":"test.json"
    }
]
for simple in data_root:
    path = simple["root"]
    instruction = "bc2gm实体识别"
    with open(path,"r",encoding="utf-8") as f:
        content = f.read()

    data = json.loads(content)
    new_data = []
    for item in data:
        sentence = item["sentence"]
        entities = []
        for entity in item["entities"]:
            entities.append({
                "text":entity["name"],
                "type":entity["type"]
            })
        if entities:
            output = "\n".join(
                f"{entity['text']}:{entity['type']}"
                for entity in entities
            )
        else:
            output = "无实体"
        new_item = {
            "instruction":instruction,
            "input":sentence,
            "output":output
        }
        new_data.append(new_item)
    out_root = f"bc_{simple['name']}.json"
    with open(out_root, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
